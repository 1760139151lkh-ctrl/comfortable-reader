function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value !== null && typeof value === 'object') {
    const result: Record<string, unknown> = Object.create(null);
    for (const key of Object.keys(value).sort()) {
      result[key] = canonicalValue((value as Record<string, unknown>)[key]);
    }
    return result;
  }
  return value;
}

/** A stable identity for an unfinished note, independent of JSON key order. */
export async function unfinishedNoteIdentity(value: unknown): Promise<string> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('未提交笔记格式无效');
  }
  const note = { ...(value as Record<string, unknown>) };
  delete note.import_id;
  const bytes = new TextEncoder().encode(JSON.stringify(canonicalValue(note)));
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return 'import-' + Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('');
}
