import type { Row } from "./api";

/** Minimal RFC 4180 CSV parser (quotes, escaped quotes, CRLF, BOM) with per-column type inference. */
export function parseCsv(text: string, maxRows = 50_000): { rows: Row[]; columns: string[]; truncated: boolean } {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const records: string[][] = [];
  let field = "", rec: string[] = [], inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else inQuotes = false; }
      else field += ch;
    } else if (ch === '"') inQuotes = true;
    else if (ch === ",") { rec.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      rec.push(field); field = "";
      if (rec.some((f) => f !== "") || rec.length > 1) records.push(rec);
      rec = [];
      if (records.length > maxRows) break;
    } else field += ch;
  }
  if (field !== "" || rec.length) { rec.push(field); records.push(rec); }
  const [header, ...body] = records;
  if (!header) return { rows: [], columns: [], truncated: false };
  const truncated = body.length > maxRows;
  const data = body.slice(0, maxRows);
  const numeric = header.map((_, c) => data.every((r) => r[c] === undefined || r[c] === "" || /^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(r[c].trim())) && data.some((r) => (r[c] ?? "") !== ""));
  const bool = header.map((_, c) => data.every((r) => r[c] === undefined || r[c] === "" || /^(true|false)$/i.test(r[c].trim())) && data.some((r) => (r[c] ?? "") !== ""));
  const rows = data.map((r) => Object.fromEntries(header.map((h, c) => {
    const v = r[c];
    if (v === undefined || v === "") return [h, null];
    return [h, numeric[c] ? Number(v) : bool[c] ? v.trim().toLowerCase() === "true" : v];
  })));
  return { rows, columns: header, truncated };
}
