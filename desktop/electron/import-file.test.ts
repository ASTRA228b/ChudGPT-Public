import { mkdtemp, open, rm, rmdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { expect, it } from "vitest";
import { MAX_IMPORT_BYTES, readImportJson } from "./import-file.js";

it("rejects an oversized import from file size before parsing it", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "chud-import-test-"));
  const file = path.join(directory, "large.json");
  const handle = await open(file, "w");
  try {
    await handle.truncate(MAX_IMPORT_BYTES + 1);
  } finally {
    await handle.close();
  }
  try {
    await expect(readImportJson(file)).rejects.toThrow("10 MB safety limit");
  } finally {
    await rm(file);
    await rmdir(directory);
  }
});

it("reads a valid small export", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "chud-import-test-"));
  const file = path.join(directory, "small.json");
  await writeFile(file, '{"version":1}', "utf8");
  try {
    await expect(readImportJson(file)).resolves.toEqual({ version: 1 });
  } finally {
    await rm(file);
    await rmdir(directory);
  }
});
