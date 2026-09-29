"""Parse SQL DDL (CREATE TABLE / ALTER TABLE ... FOREIGN KEY) into a RelationshipGraph."""

from __future__ import annotations

import re

from sdp.relational.schema import Column, ForeignKey, RelationshipGraph, Table

_IDENT = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[\w$.]+)'


def _unq(s: str) -> str:
    s = s.strip()
    if s and s[0] in '"`[':
        s = s[1:-1]
    return s.split(".")[-1]


def _dtype(sql_type: str) -> str:
    t = sql_type.lower()
    if re.search(r"bool|bit", t):
        return "bool"
    if re.search(r"int|serial", t):
        return "int"
    if re.search(r"float|double|real|numeric|decimal|money", t):
        return "float"
    if re.search(r"date|time", t):
        return "datetime"
    return "str"


def _split_top_level(body: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def _cols(text: str) -> list[str]:
    return [_unq(c) for c in re.findall(_IDENT, text)]


def parse_ddl(ddl: str) -> RelationshipGraph:
    ddl = re.sub(r"--[^\n]*", "", ddl)
    ddl = re.sub(r"/\*.*?\*/", "", ddl, flags=re.S)
    tables: list[Table] = []
    fk_specs: list[tuple[str, list[str], str, list[str] | None]] = []
    unique_cols: dict[str, list[list[str]]] = {}

    for m in re.finditer(rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?({_IDENT})\s*\(", ddl, re.I):
        name = _unq(m.group(1))
        depth, i = 1, m.end()
        while i < len(ddl) and depth:
            depth += {"(": 1, ")": -1}.get(ddl[i], 0)
            i += 1
        cols: list[Column] = []
        pk: list[str] = []
        unique_cols[name] = []
        for item in _split_top_level(ddl[m.end():i - 1]):
            up = item.upper()
            if re.match(r"(CONSTRAINT\s+\S+\s+)?PRIMARY\s+KEY", up):
                pk = _cols(re.search(r"\(([^)]*)\)", item).group(1))
            elif re.match(r"(CONSTRAINT\s+\S+\s+)?FOREIGN\s+KEY", up):
                fm = re.search(rf"FOREIGN\s+KEY\s*\(([^)]*)\)\s*REFERENCES\s+({_IDENT})\s*(?:\(([^)]*)\))?", item, re.I)
                fk_specs.append((name, _cols(fm.group(1)), _unq(fm.group(2)), _cols(fm.group(3)) if fm.group(3) else None))
            elif re.match(r"(CONSTRAINT\s+\S+\s+)?UNIQUE", up):
                unique_cols[name].append(_cols(re.search(r"\(([^)]*)\)", item).group(1)))
            elif re.match(r"(KEY|INDEX|CHECK|CONSTRAINT)\b", up):
                continue
            else:
                cm = re.match(rf"({_IDENT})\s+([A-Za-z_]+(?:\s*\([^)]*\))?)(.*)", item, re.S)
                if not cm:
                    continue
                cname, ctype, rest = _unq(cm.group(1)), cm.group(2), cm.group(3)
                restu = rest.upper()
                is_pk = "PRIMARY KEY" in restu
                cols.append(Column(name=cname, dtype=_dtype(ctype), nullable=not (is_pk or "NOT NULL" in restu)))
                if is_pk:
                    pk = [cname]
                if "UNIQUE" in restu:
                    unique_cols[name].append([cname])
                rm = re.search(rf"REFERENCES\s+({_IDENT})\s*(?:\(([^)]*)\))?", rest, re.I)
                if rm:
                    fk_specs.append((name, [cname], _unq(rm.group(1)), _cols(rm.group(2)) if rm.group(2) else None))
        tables.append(Table(name=name, columns=cols, primary_key=pk))

    for am in re.finditer(
        rf"ALTER\s+TABLE\s+(?:ONLY\s+)?({_IDENT})\s+ADD\s+(?:CONSTRAINT\s+\S+\s+)?FOREIGN\s+KEY\s*\(([^)]*)\)\s*"
        rf"REFERENCES\s+({_IDENT})\s*(?:\(([^)]*)\))?", ddl, re.I):
        fk_specs.append((_unq(am.group(1)), _cols(am.group(2)), _unq(am.group(3)), _cols(am.group(4)) if am.group(4) else None))

    graph = RelationshipGraph(tables=tables)
    by_name = {t.name: t for t in tables}
    for child, ccols, parent, pcols in fk_specs:
        if parent not in by_name or child not in by_name:
            raise ValueError(f"FK {child}->{parent} references an unknown table")
        pcols = pcols or by_name[parent].primary_key
        ct = by_name[child]
        one_to_one = ct.primary_key == ccols or ccols in unique_cols.get(child, [])
        graph.foreign_keys.append(ForeignKey(
            child_table=child, child_columns=ccols, parent_table=parent, parent_columns=pcols,
            cardinality="1:1" if one_to_one else "1:N",
            nullable=any(ct.column(c).nullable for c in ccols), source="ddl"))
    return graph
