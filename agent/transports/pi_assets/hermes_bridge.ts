/**
 * Hermes-owned policy/approval extension for Pi 0.84.3.
 *
 * This is defense in depth, not containment. Write-capable sessions run the
 * entire Pi process inside the separately enforced Docker boundary.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { lstatSync, readlinkSync, realpathSync } from "node:fs";
import { dirname, isAbsolute, parse, relative, resolve, sep } from "node:path";

const WORKSPACE = process.env.HERMES_PI_WORKSPACE || process.cwd();
const WORKSPACE_REAL = realpathSync(WORKSPACE);
const WORKSPACE_GIT = resolve(WORKSPACE_REAL, ".git");

function resolveSymlinks(candidate: string, followed = 0): string {
  if (followed > 40) throw new Error("too many symbolic links in mediated path");
  const absolute = resolve(candidate);
  const root = parse(absolute).root;
  const tail = relative(root, absolute);
  const parts = tail ? tail.split(sep).filter(Boolean) : [];
  let current = root;
  for (let index = 0; index < parts.length; index += 1) {
    const next = resolve(current, parts[index]);
    try {
      const metadata = lstatSync(next);
      if (metadata.isSymbolicLink()) {
        const link = readlinkSync(next);
        const target = resolve(dirname(next), link, ...parts.slice(index + 1));
        return resolveSymlinks(target, followed + 1);
      }
      current = next;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      // The existing prefix is already canonical. Resolve the absent suffix
      // from it so dangling leaf and parent symlinks cannot hide traversal.
      return resolve(current, ...parts.slice(index));
    }
  }
  return current;
}

function mediatedRealpath(candidate: string): string {
  return resolveSymlinks(resolve(WORKSPACE_REAL, candidate));
}

function isSameOrDescendant(root: string, candidate: string): boolean {
  const rel = relative(root, candidate);
  return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
}

export function canonicalInside(candidate: string): boolean {
  const lexical = resolve(WORKSPACE_REAL, candidate);
  if (!isSameOrDescendant(WORKSPACE_REAL, lexical)) return false;
  // Deny the linked-worktree pointer itself and every lexical descendant
  // before following symlinks. A .git file is host control-plane metadata,
  // not project content, and disclosing it exposes the common-worktree path.
  if (isSameOrDescendant(WORKSPACE_GIT, lexical)) return false;
  try {
    const canonical = mediatedRealpath(candidate);
    if (!isSameOrDescendant(WORKSPACE_REAL, canonical)) return false;
    // An ordinary in-workspace symlink may alias .git. Compare the resolved
    // target too, so aliases cannot read, edit, or overwrite the pointer.
    const gitCanonical = resolveSymlinks(WORKSPACE_GIT);
    return !isSameOrDescendant(gitCanonical, canonical);
  } catch {
    // ENOTDIR and malformed/symlink-loop cases are denied, never delegated to
    // a tool implementation whose path semantics could differ.
    return false;
  }
}

function pathArguments(input: Record<string, unknown>): string[] {
  return [input.path, input.file, input.directory, input.cwd]
    .filter((value): value is string => typeof value === "string" && value.length > 0);
}

export default function hermesBridge(pi: ExtensionAPI) {
  pi.on("project_trust", async () => ({ trusted: "no", remember: false }));

  pi.on("before_agent_start", async event => ({
    systemPrompt:
      event.systemPrompt +
      "\n\nHermes owns workspace assignment, approvals, and final verification. " +
      "Stay inside the assigned workspace. Never invoke coding_task recursively. " +
      "This extension is policy mediation; the Docker runtime is the containment boundary."
  }));

  pi.on("tool_call", async event => {
    const input = (event.input || {}) as Record<string, unknown>;
    for (const candidate of pathArguments(input)) {
      if (!canonicalInside(candidate)) {
        return { block: true, reason: "Path is outside the Hermes-assigned workspace", terminate: true };
      }
    }
    if (event.toolName === "bash" || event.toolName === "powershell") {
      return { block: true, reason: "Model shell tools are disabled by Hermes", terminate: true };
    }
  });
}
