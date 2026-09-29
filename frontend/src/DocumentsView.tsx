import { useEffect, useState } from "react";
import { get, post, postBlob, saveBlob, type DocLayout, type DocPreview, type DocTypes, type LocaleInfo } from "./api";
import { BulkPanel } from "./BulkPanel";
import { PreviewFrame, Skeleton } from "./Preview";
import { ScanPanel } from "./ScanPanel";
import { VisualizePanel } from "./viz/LazyVisualize";
import { localeName, LOCALE_HELP } from "./locales";
import { Field, Panel, Tabs, useLive } from "./ui";

const TABS = ["Preview", "Visualize", "PDF page", "Scan", "Bulk"] as const;
interface QueryParse { ok: boolean; filter: Record<string, string | number | null>; unparsed: string[] }
type DocType = string;

/** HTML rendering of the same Layout the PDF uses. `dir` + logical CSS properties mirror it for RTL locales. */
function LayoutPreview({ layout }: { layout: DocLayout }) {
  const rtl = layout.direction === "rtl";
  return (
    <article className="docpage" dir={layout.direction} lang={rtl ? "ar" : undefined}>
      <div className="dp-head">
        <div>{layout.header_start.map((h, i) => <div key={i} style={{ fontWeight: h.bold && i === 0 ? 700 : 400 }} className={i ? "muted" : ""}>{h.text}</div>)}</div>
        <div className="end"><div className="dp-title">{layout.title}</div>{layout.header_end.map((t) => <div key={t} className="muted">{t}</div>)}<div><b>{layout.number}</b></div></div>
      </div>
      {layout.block_label && <div style={{ marginBottom: 12 }}><div className="muted">{layout.block_label}</div>{layout.block_lines.map((l) => <div key={l}>{l}</div>)}</div>}
      <div className="tablewrap">
        <table>
          <thead><tr>{layout.columns.map((c, i) => <th key={i} className={c.align === "end" ? "end" : ""} style={{ width: `${c.width * 100}%` }}>{c.label}</th>)}</tr></thead>
          <tbody>{layout.rows.map((r, i) => <tr key={i}>{r.map((cell, j) => <td key={j} dir="auto" className={layout.columns[j].align === "end" ? "end" : ""}>{cell}</td>)}</tr>)}</tbody>
        </table>
      </div>
      <div className="dp-totals">{layout.totals.map((t) => <div key={t.label} className={t.strong ? "strong" : ""}><span>{t.label}</span><span dir="ltr">{t.value}</span></div>)}</div>
      {layout.notes.map((n) => <p key={n} className="muted">{n}</p>)}
    </article>
  );
}

export function DocumentsWorkspace() {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Preview");
  const [docType, setDocType] = useState<DocType>("invoice");
  const [locale, setLocale] = useState("en-US");
  const [region, setRegion] = useState("");
  const [seed, setSeed] = useState(1);
  const [size, setSize] = useState(5);
  const [inclusive, setInclusive] = useState(false);
  const [native, setNative] = useState(false);
  const [digits, setDigits] = useState(false);
  const [bom, setBom] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [pdfErr, setPdfErr] = useState<string | null>(null);
  const [pdfBusy, setPdfBusy] = useState(false);
  const [query, setQuery] = useState("");
  const locales = useLive(() => get<LocaleInfo[]>("/api/locales"), [], 0).data;
  const types = useLive(() => get<DocTypes>("/api/documents/types"), [], 0).data;
  const typeInfo = types?.details.find((d) => d.name === docType);
  const templated = typeInfo?.engine === "html";   // template documents: Latin script, no native/tax-inclusive options
  const info = locales?.find((l) => l.code === locale);
  const parsed = useLive((s) => (docType === "statement" && query.trim() ? post<QueryParse>("/api/documents/statement/parse", { query }, s) : Promise.resolve(null)), [docType, query], 400);
  const queryOk = !query.trim() || (parsed.data?.ok ?? false);

  const spec = {
    locale, region: region || null, seed, ...(templated ? {} : { native, native_digits: native && digits }),
    ...(docType === "invoice" ? { n_lines: size, tax_inclusive: inclusive } : docType === "receipt" ? { n_lines: size }
      : docType === "statement" ? { n_transactions: size * 10, days: 365, ...(query.trim() && queryOk ? { query } : {}) }
        : typeInfo?.params.includes("n_lines") ? { n_lines: size } : {}),
  };
  const req = { doc_type: docType, spec };
  const preview = useLive((s) => post<DocPreview>("/api/documents/preview", req, s), [JSON.stringify(req)]);

  // rendered PDF page, debounced, only while that tab is open
  useEffect(() => {
    if (tab !== "PDF page") return;
    let cancelled = false;
    setPdfBusy(true);
    const t = setTimeout(async () => {
      try {
        const blob = await postBlob("/api/documents/pdf", req);
        if (cancelled) return;
        setPdfUrl((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(blob); });
        setPdfErr(null);
      } catch (e) {
        if (!cancelled) { let m = (e as Error).message; try { const j = JSON.parse(m); m = j.detail?.message ?? m; } catch { /* plain text */ } setPdfErr(m); }
      } finally { if (!cancelled) setPdfBusy(false); }
    }, 500);
    return () => { cancelled = true; clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, JSON.stringify(req)]);

  const dl = async (kind: "pdf" | "json") => {
    setErr(null);
    try { saveBlob(await postBlob(`/api/documents/${kind}`, { ...req, bom }), `${docType}.${kind}`); } catch (e) { setErr((e as Error).message); }
  };
  const warn = preview.data && (preview.data.document as { presentation?: { notes?: string[] } }).presentation?.notes;

  return (
    <>
      <main className="center">
        <h1>Documents</h1>
        <p className="lede">Invoices, receipts and statements in any supported language. Amounts are exact decimals and every total reconciles.</p>
        <Tabs tabs={TABS} value={tab} onChange={setTab} />
        {err && <div className="err" role="alert">{err}</div>}
        {tab === "Preview" && (
          <PreviewFrame loading={preview.busy} error={preview.error} hasData={!!preview.data} emptyTitle="No document yet" skeleton={<Skeleton rows={9} cols={4} />}>
            {preview.data && (
              <>
                {warn?.map((w) => <p key={w} className="muted">⚠ {w}</p>)}
                {preview.data.reconciliation_errors.length > 0 && <div className="err">Totals do not reconcile: {preview.data.reconciliation_errors.join("; ")}</div>}
                {preview.data.layout ? <LayoutPreview layout={preview.data.layout} /> : (
                  <iframe className="htmlframe" title="Document preview" sandbox="" srcDoc={preview.data.html ?? ""} />
                )}
                <div style={{ marginTop: 16 }}>
                  <button className="btn primary" onClick={() => dl("pdf")}>Download PDF</button>{" "}
                  <button className="btn" onClick={() => dl("json")}>Download JSON ground truth</button>{" "}
                  {docType === "statement" && <button className="btn" onClick={async () => { try { saveBlob(await postBlob("/api/documents/csv", { ...req, bom }), "statement.csv"); } catch (e) { setErr((e as Error).message); } }}>Download CSV</button>}
                </div>
              </>
            )}
          </PreviewFrame>
        )}
        {tab === "PDF page" && (
          <PreviewFrame loading={pdfBusy} error={pdfErr} hasData={!!pdfUrl} emptyTitle="Rendering the page…" skeleton={<Skeleton rows={12} cols={2} />}>
            {pdfUrl && <iframe className="pdfframe" title="Rendered PDF page" src={`${pdfUrl}#toolbar=0&view=FitH`} />}
          </PreviewFrame>
        )}
        {tab === "Visualize" && <VisualizePanel datasetId="documents-demo" kind="documents" />}
        {tab === "Scan" && <ScanPanel docType={docType} spec={spec} />}
        {tab === "Bulk" && <BulkPanel docType={docType} spec={spec} seed={seed} />}
      </main>
      <aside className="config">
        <h2>Settings</h2>
        <Field label="Document type"><select value={docType} onChange={(e) => setDocType(e.target.value)}>
          {(types?.details ?? [{ name: "invoice", title: "Invoice", engine: "reportlab", params: [] }]).map((d) => <option key={d.name} value={d.name}>{d.title || d.name}{d.engine === "html" ? " · template" : ""}</option>)}</select></Field>
        <Field label="Country and language" hint={LOCALE_HELP}><select value={locale} onChange={(e) => { setLocale(e.target.value); setRegion(""); }}>
          {(locales ?? [{ code: "en-US" } as LocaleInfo]).map((l) => <option key={l.code} value={l.code}>{localeName(l.code, l.direction === "rtl")}</option>)}</select></Field>
        {(!templated || typeInfo?.params.includes("n_lines")) && <Field label={docType === "statement" ? "Size (×10 transactions)" : "Line items"}><input type="number" min={1} max={100} value={size} onChange={(e) => setSize(Math.min(100, Math.max(1, +e.target.value || 1)))} /></Field>}
        <Panel title="Language and script" badge={native ? "on" : null}>
        {templated
          ? <p className="muted">Template documents render in Latin script.</p>
          : <label className="field"><span>Write the document in the local language and script{info?.complex_shaping ? " (not available for this script: English labels are used)" : ""}</span>
            <input type="checkbox" checked={native} onChange={(e) => setNative(e.target.checked)} /></label>}
        {!templated && native && info?.native_digits && <label className="field"><span>Native digits (٠١٢ / ۰۱۲)</span><input type="checkbox" checked={digits} onChange={(e) => setDigits(e.target.checked)} /></label>}
        </Panel>
        <Panel title={docType === "statement" ? "Filter transactions" : "Tax and prices"} badge={docType === "statement" ? (query.trim() ? "on" : null) : (region || inclusive ? "set" : null)} defaultOpen={docType === "statement"}>
        {docType === "statement" && (
          <Field label="Query, e.g. last 90 days, balance over $500">
            <input value={query} placeholder="last 90 days, debits only over 50" onChange={(e) => setQuery(e.target.value)} />
            {query.trim() && parsed.data && (parsed.data.ok
              ? <div className="muted" style={{ marginTop: 4 }}>Understood: {Object.entries(parsed.data.filter).filter(([k, v]) => v && !["raw", "unparsed"].includes(k) && !(Array.isArray(v) && !v.length)).map(([k, v]) => `${k.replace("_", " ")} ${v}`).join(" · ")}</div>
              : <div className="err">Couldn’t understand: {parsed.data.unparsed.join(" ")}</div>)}
            {parsed.error && <div className="err">{parsed.error}</div>}
          </Field>
        )}
        {docType !== "statement" && <Field label="Tax region (blank = default rate)"><input value={region} placeholder="e.g. CA, SINDH, KA" onChange={(e) => setRegion(e.target.value.trim())} /></Field>}
        {docType === "invoice" && <label className="field"><span>Prices include tax</span><input type="checkbox" checked={inclusive} onChange={(e) => setInclusive(e.target.checked)} /></label>}
        </Panel>
        <Panel title="Advanced">
          <Field label="Repeat code" hint="Same number gives exactly the same document again."><input type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} /></Field>
          <label className="field"><span>UTF-8 with BOM (JSON download)</span><input type="checkbox" checked={bom} onChange={(e) => setBom(e.target.checked)} /></label>
        </Panel>
      </aside>
    </>
  );
}
