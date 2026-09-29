import { useEffect, useState } from "react";
import { get, post, postBlob, saveBlob, type LocaleInfo, type Row } from "./api";
import { PreviewFrame, Skeleton } from "./Preview";
import { localeName, LOCALE_HELP } from "./locales";
import { DataTable, Field, Panel, Tabs, useLive } from "./ui";

const TABS = ["Coherent people", "Name variants", "Code-mixed text"] as const;

interface People { columns: string[]; preview: Row[]; coherence: { coherent_pct: number; by_check: Record<string, { checked: number; failed: number }>; mix: Record<string, number> } }
interface Entities { columns: string[]; preview: Row[]; entities: number; records: number; matching_pairs: { positive: number; negative: number } }
interface CodeMix { preview: { text: string; realized_mix: number | null; tokens: [string, string][] }[]; mean_english_token_ratio: number }
interface Options { pairs: { name: string; label: string; topics: string[] }[] }

export function LocaleLab() {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Coherent people");
  const [seed, setSeed] = useState(0);
  const [bom, setBom] = useState(true);
  const [mix, setMix] = useState("70% Pakistani, 30% American");
  const [n, setN] = useState(200);
  const [locale, setLocale] = useState("ur-PK");
  const [variants, setVariants] = useState(5);
  const [confusers, setConfusers] = useState(0.1);
  const [typos, setTypos] = useState(false);
  const [pair, setPair] = useState("roman_urdu");
  const [topic, setTopic] = useState("banking");
  const [level, setLevel] = useState(0.5);
  const [err, setErr] = useState<string | null>(null);
  const locales = useLive(() => get<LocaleInfo[]>("/api/locales"), [], 0).data;
  const options = useLive(() => get<Options>("/api/locale/codemix/options"), [], 0).data;
  useEffect(() => { const t = options?.pairs.find((p) => p.name === pair)?.topics; if (t && !t.includes(topic)) setTopic(t[0]); }, [pair, options, topic]);

  const people = useLive((s) => post<People>("/api/locale/people", { n, mix, seed }, s), [tab === "Coherent people", n, mix, seed], 400);
  const ents = useLive((s) => post<Entities>("/api/locale/entities", { n: Math.min(n, 300), locale, variants, confusers, typos, seed }, s), [tab === "Name variants", n, locale, variants, confusers, typos, seed], 400);
  const cm = useLive((s) => post<CodeMix>("/api/locale/codemix", { n: 30, pair, topic, level, seed }, s), [tab === "Code-mixed text", pair, topic, level, seed], 300);

  const download = async (path: string, body: object, name: string) => {
    setErr(null);
    try { saveBlob(await postBlob(path, { ...body, download: true, bom }), name); } catch (e) { setErr((e as Error).message); }
  };
  const chosen = tab === "Coherent people" ? people : tab === "Name variants" ? ents : cm;

  return (
    <>
      <main className="center">
        <h1>Locale lab</h1>
        <p className="lede">Culturally coherent records, the same person in several scripts and spellings, and code-mixed text. All names and numbers are fictional.</p>
        <Tabs tabs={TABS} value={tab} onChange={setTab} />
        {err && <div className="err" role="alert">{err}</div>}
        <PreviewFrame loading={chosen.busy} error={chosen.error} hasData={!!chosen.data} emptyTitle="Nothing generated yet" skeleton={<Skeleton rows={7} cols={6} />}>
          {tab === "Coherent people" && people.data && (
            <>
              <div className="cards">
                <div className="card"><div className="big">{people.data.coherence.coherent_pct.toFixed(0)}%</div><div className="label">rows coherent in their own locale</div></div>
                {Object.entries(people.data.coherence.mix).map(([k, v]) => <div className="card" key={k}><div className="big">{(v * 100).toFixed(0)}%</div><div className="label">{k}</div></div>)}
              </div>
              <p className="muted">Every row comes from one locale context: the phone, national-ID checksum, name script, city, email host and currency all match. Checked: {Object.keys(people.data.coherence.by_check).join(", ")}.</p>
              <DataTable columns={people.data.columns} rows={people.data.preview} />
              <button className="btn" style={{ marginTop: 16 }} onClick={() => download("/api/locale/people", { n, mix, seed }, "people.csv")}>Download {n.toLocaleString()} rows (CSV)</button>
            </>
          )}
          {tab === "Name variants" && ents.data && (
            <>
              <div className="cards">
                <div className="card"><div className="big">{ents.data.entities}</div><div className="label">entities (shared entity_id)</div></div>
                <div className="card"><div className="big">{ents.data.records}</div><div className="label">name records</div></div>
                <div className="card"><div className="big">{ents.data.matching_pairs.positive} / {ents.data.matching_pairs.negative}</div><div className="label">match / non-match pairs (with hard negatives)</div></div>
              </div>
              <DataTable columns={["entity_id", "name", "script", "style", "canonical", "dob", "confuser_of"]} rows={ents.data.preview} rowMark={(i) => (ents.data!.preview[i].confuser_of ? "different person, same name" : undefined)} />
              <button className="btn" style={{ marginTop: 16 }} onClick={() => download("/api/locale/entities", { n: Math.min(n, 300), locale, variants, confusers, typos, seed }, "entities.csv")}>Download (CSV)</button>
            </>
          )}
          {tab === "Code-mixed text" && cm.data && (
            <>
              <div className="cards">
                <div className="card"><div className="big">{(cm.data.mean_english_token_ratio * 100).toFixed(0)}%</div><div className="label">of all tokens are English. The {Math.round(level * 100)}% level switches that share of content words; grammar words stay in the base language.</div></div>
              </div>
              <div className="legend"><span><i className="sw syn" /> English</span><span><i className="sw real" /> base language</span></div>
              {cm.data.preview.map((r, i) => (
                <p key={i} className="cm-line">{r.tokens.map(([t, lang], j) => <span key={j} className={`tok ${lang}`}>{j > 0 && lang !== "punct" ? " " : ""}{t}</span>)}</p>
              ))}
              <p className="muted">Templates are hand-written and small; have a fluent speaker review them before using the text as evidence of how people write.</p>
              <button className="btn" onClick={() => download("/api/locale/codemix", { n: 500, pair, topic, level, seed }, "code-mixed.csv")}>Download 500 sentences with token language tags (CSV)</button>
            </>
          )}
        </PreviewFrame>
      </main>
      <aside className="config">
        <h2>Settings</h2>
        {tab !== "Code-mixed text" && <Field label={tab === "Name variants" ? "Entities (max 300)" : "Rows"}><input type="number" min={1} max={tab === "Name variants" ? 300 : 20000} value={n} onChange={(e) => setN(Math.max(1, +e.target.value || 1))} /></Field>}
        {tab === "Coherent people" && <Field label="Mix of countries" hint="Write percentages and country or language names, for example: 70% Pakistani, 30% American. Codes like ur-PK also work."><input value={mix} onChange={(e) => setMix(e.target.value)} placeholder="70% Pakistani, 30% American" /></Field>}
        {tab === "Name variants" && (
          <>
            <Field label="Country and language" hint={LOCALE_HELP}><select value={locale} onChange={(e) => setLocale(e.target.value)}>{(locales ?? [{ code: "ur-PK" } as LocaleInfo]).map((l) => <option key={l.code} value={l.code}>{localeName(l.code)}</option>)}</select></Field>
            <div className="row2">
              <Field label="Variants per entity"><input type="number" min={2} max={12} value={variants} onChange={(e) => setVariants(Math.min(12, Math.max(2, +e.target.value || 2)))} /></Field>
              <Field label="Confusers (0–0.5)"><input type="number" min={0} max={0.5} step={0.05} value={confusers} onChange={(e) => setConfusers(Math.min(0.5, Math.max(0, +e.target.value || 0)))} /></Field>
            </div>
            <label className="field"><span>Add typo variants</span><input type="checkbox" checked={typos} onChange={(e) => setTypos(e.target.checked)} /></label>
          </>
        )}
        {tab === "Code-mixed text" && (
          <>
            <Field label="Language pair"><select value={pair} onChange={(e) => setPair(e.target.value)}>{options?.pairs.map((p) => <option key={p.name} value={p.name}>{p.label}</option>)}</select></Field>
            <Field label="Topic"><select value={topic} onChange={(e) => setTopic(e.target.value)}>{options?.pairs.find((p) => p.name === pair)?.topics.map((t) => <option key={t}>{t}</option>)}</select></Field>
            <Field label={`Mixing level: ${Math.round(level * 100)}% English`}><input type="range" min={0} max={1} step={0.05} value={level} onChange={(e) => setLevel(+e.target.value)} /></Field>
          </>
        )}
        <Panel title="Advanced">
          <Field label="Repeat code" hint="Same number gives exactly the same data again."><input type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} /></Field>
          <label className="field"><span>UTF-8 BOM in CSV downloads (helps Excel with Urdu, Arabic, Chinese)</span><input type="checkbox" checked={bom} onChange={(e) => setBom(e.target.checked)} /></label>
        </Panel>
      </aside>
    </>
  );
}
