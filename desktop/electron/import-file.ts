import { open } from "node:fs/promises";

export const MAX_IMPORT_BYTES = 10_000_000;

export async function readImportJson(filePath: string): Promise<unknown> {
  const handle = await open(filePath, "r");
  try {
    // Reject before reading; character count is not a byte limit.
    if ((await handle.stat()).size > MAX_IMPORT_BYTES)
      throw new Error("Import is larger than the 10 MB safety limit.");
    const raw = await handle.readFile("utf8");
    if (Buffer.byteLength(raw, "utf8") > MAX_IMPORT_BYTES)
      throw new Error("Import is larger than the 10 MB safety limit.");
    return JSON.parse(raw) as unknown;
  } finally {
    await handle.close();
  }
}
