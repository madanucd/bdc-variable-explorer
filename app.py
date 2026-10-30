"""
BDC Variable Explorer — browse BioData Catalyst (dbGaP) variables by study,
dataset, unit, value and ontology term, and collect them into a cohort.

Run with:
    streamlit run app.py

Needs bdc.db next to this file (built by parse_owl.py).
"""
import sqlite3
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).parent / "bdc.db"
# Optional images for the About tab; each is skipped if the file isn't there.
ARCHITECTURE_IMG = Path(__file__).parent / "architecture.svg"
TISLAB_LOGO = Path(__file__).parent / "tislab_logo.png"
UNC_LOGO = Path(__file__).parent / "unc_logo.png"
PAGE_SIZE = 25          # rows per page in variable lists
COHORT_PAGE_SIZE = 50   # rows per page, per dataset, in the Cohort tab
# Caches below are shared by all users. Per-selection caches are capped so they
# can't grow without bound on a long-running server.
SEL_CACHE = dict(max_entries=200, ttl=3600)
DBGAP_VAR_URL = "https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/variable.cgi?study_id={study}&phv={phv}"
DBGAP_STUDY_URL = "https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id={study}"

st.set_page_config(page_title="BDC Variable Explorer", page_icon="◈", layout="wide")

# Keep images (e.g. the architecture diagram) within the screen width on smaller screens.
st.markdown("<style>img { max-width: 100% !important; height: auto !important; }</style>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data access (read-only; one connection per Streamlit thread)
# ---------------------------------------------------------------------------
_local = threading.local()


def get_conn():
    conn = getattr(_local, "conn", None)
    if conn is None:
        # Read-only + immutable: works on read-only mounts (Streamlit Cloud), writes
        # nothing next to bdc.db, and skips file locking between concurrent users.
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro&immutable=1", uri=True,
                               check_same_thread=False)
        _local.conn = conn
    return conn


def q(sql, params=()):
    return pd.read_sql(sql, get_conn(), params=list(params))


# Columns every variable list carries (one row per variable)
VAR_COLS = """c.uri, c.label, c.comment, c.study_name, c.study_accession,
              c.variable_id, c.unit, c.n,
              (SELECT group_concat(DISTINCT d.dataset_name) FROM cde_dataset d
                WHERE d.cde_uri = c.uri) AS datasets"""


@st.cache_data
def load_counts():
    one = lambda sql: int(q(sql).iloc[0, 0])
    return {
        "variables": one("SELECT COUNT(*) FROM cde"),
        "studies":   one("SELECT COUNT(DISTINCT study_accession) FROM cde"),
        "datasets":  one("SELECT COUNT(DISTINCT dataset_id) FROM cde_dataset"),
        "units":     one("SELECT COUNT(DISTINCT unit) FROM cde WHERE unit IS NOT NULL AND unit != ''"),
        "onto":      one("SELECT COUNT(*) FROM onto_term"),
        "bundles":   one("SELECT COUNT(*) FROM bundle"),
        "cdm":       one("SELECT COUNT(*) FROM cdm"),
    }


@st.cache_data
def load_studies():
    return q("""SELECT c.study_accession, MAX(c.study_name) AS study_name,
                       COUNT(*) AS n_vars,
                       (SELECT COUNT(DISTINCT dataset_id) FROM cde_dataset d
                         WHERE d.study_accession = c.study_accession) AS n_datasets
                FROM cde c WHERE c.study_accession IS NOT NULL
                GROUP BY c.study_accession ORDER BY study_name""")


@st.cache_data
def load_datasets(study_accession=None):
    sql = """SELECT dataset_id, dataset_name, study_accession,
                    COUNT(DISTINCT cde_uri) AS n_vars
             FROM cde_dataset {where} GROUP BY dataset_id, dataset_name, study_accession
             ORDER BY dataset_name"""
    if study_accession:
        return q(sql.format(where="WHERE study_accession = ?"), [study_accession])
    return q(sql.format(where=""))


@st.cache_data
def load_units():
    return q("""SELECT unit, COUNT(*) AS n_vars FROM cde
                WHERE unit IS NOT NULL AND unit != '' GROUP BY unit ORDER BY unit COLLATE NOCASE""")


@st.cache_data
def load_ontologies():
    return q("SELECT DISTINCT ontology FROM onto_term ORDER BY 1")["ontology"].tolist()


@st.cache_data(**SEL_CACHE)
def search_variables(text, study, dataset_id, unit, ontology, limit=500):
    sql = f"SELECT DISTINCT {VAR_COLS} FROM cde c"
    joins, where, params = [], [], []
    if dataset_id:
        joins.append("JOIN cde_dataset ds ON ds.cde_uri = c.uri AND ds.dataset_id = ?")
        params.append(dataset_id)
    if ontology:
        joins.append("JOIN cde_onto co ON co.cde_uri = c.uri "
                     "JOIN onto_term ot ON ot.uri = co.onto_uri AND ot.ontology = ?")
        params.append(ontology)
    if study:
        where.append("c.study_accession = ?")
        params.append(study)
    if unit:
        where.append("c.unit = ?")
        params.append(unit)
    if text:
        where.append("(c.label LIKE ? OR c.comment LIKE ? OR c.variable_id LIKE ?)")
        params += [f"%{text}%"] * 3
    sql += " " + " ".join(joins)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY c.label LIMIT {int(limit)}"
    return q(sql, params)


@st.cache_data(**SEL_CACHE)
def variables_in_dataset(dataset_id):
    return q(f"""SELECT {VAR_COLS}, ds.variable_id AS dataset_variable_id
                 FROM cde_dataset ds JOIN cde c ON c.uri = ds.cde_uri
                 WHERE ds.dataset_id = ? ORDER BY c.label""", [dataset_id])


@st.cache_data(**SEL_CACHE)
def variables_with_unit(unit):
    return q(f"SELECT {VAR_COLS} FROM cde c WHERE c.unit = ? ORDER BY c.study_name, c.label", [unit])


@st.cache_data
def load_value_options():
    return q("SELECT value_label, COUNT(DISTINCT cde_uri) n FROM cde_value GROUP BY value_label ORDER BY value_label")


@st.cache_data(**SEL_CACHE)
def variables_with_value(value_label):
    return q(f"""SELECT DISTINCT {VAR_COLS} FROM cde_value v JOIN cde c ON c.uri = v.cde_uri
                 WHERE v.value_label = ? ORDER BY c.label""", [value_label])


@st.cache_data
def load_onto_df():
    return q("""SELECT t.*, (SELECT COUNT(DISTINCT cde_uri) FROM cde_onto o WHERE o.onto_uri = t.uri) n_vars
                FROM onto_term t ORDER BY ontology, term_id""")


@st.cache_data(**SEL_CACHE)
def variables_for_term(onto_uri):
    return q(f"""SELECT {VAR_COLS}, group_concat(DISTINCT co.via_value) AS via_value
                 FROM cde_onto co JOIN cde c ON c.uri = co.cde_uri
                 WHERE co.onto_uri = ? GROUP BY c.uri ORDER BY c.label""", [onto_uri])


@st.cache_data(**SEL_CACHE)
def get_variable_detail(uri):
    cde = q("SELECT * FROM cde WHERE uri = ?", [uri])
    datasets = q("""SELECT dataset_name, dataset_id, dataset_version, variable_id, study_accession
                    FROM cde_dataset WHERE cde_uri = ? ORDER BY dataset_name""", [uri])
    values = q("SELECT value_label FROM cde_value WHERE cde_uri = ?", [uri])
    onto = q("""SELECT ot.ontology, ot.term_id, ot.label, ot.ols_url, ot.bioportal_url, co.via_value
                FROM cde_onto co JOIN onto_term ot ON ot.uri = co.onto_uri
                WHERE co.cde_uri = ? ORDER BY ot.ontology, ot.term_id""", [uri])
    bundles = q("""SELECT b.label, b.kind, cb.strength FROM cde_bundle cb
                   JOIN bundle b ON b.uri = cb.bundle_uri WHERE cb.cde_uri = ?""", [uri])
    cdm = q("""SELECT m.cdm_name, m.table_name, m.column_name, cc.strength FROM cde_cdm cc
               JOIN cdm m ON m.uri = cc.cdm_uri WHERE cc.cde_uri = ?""", [uri])
    return cde, datasets, values, onto, bundles, cdm


def cohort_rows(uris):
    """One row per (variable, dataset) for the cohort tab, in study -> dataset -> variable order.
    Not cached: the key would be the whole cohort, so every +/− would store a new copy."""
    if not uris:
        return pd.DataFrame()
    marks = ",".join("?" * len(uris))
    return q(f"""SELECT c.uri, c.label, c.comment, c.study_name, c.study_accession, c.unit, c.n,
                        COALESCE(ds.dataset_name, '(no dataset)') AS dataset_name,
                        ds.dataset_id, COALESCE(ds.variable_id, c.variable_id) AS variable_id
                 FROM cde c LEFT JOIN cde_dataset ds ON ds.cde_uri = c.uri
                 WHERE c.uri IN ({marks})
                 ORDER BY c.study_name, dataset_name, c.label""", list(uris))


# ---------------------------------------------------------------------------
# Cohort state (per browser session)
# ---------------------------------------------------------------------------
if "cohort" not in st.session_state:
    st.session_state.cohort = {}          # uri -> None ; a dict keeps insertion order


def cohort_add(*uris):
    for u in uris:
        st.session_state.cohort[u] = None


def cohort_remove(*uris):
    for u in uris:
        st.session_state.cohort.pop(u, None)


def cohort_clear():
    st.session_state.cohort = {}


def fmt_n(n):
    return "—" if n is None or pd.isna(n) else f"{int(n):,}"


def phv_link(variable_id, study_accession):
    """phv00112421.v2.p2 -> markdown link to the dbGaP variable page."""
    if not variable_id:
        return ""
    first = str(variable_id).split(",")[0].strip()
    num = "".join(ch for ch in first.split(".")[0] if ch.isdigit())
    if not num or not study_accession:
        return f"`{first}`"
    return f"[`{first}`]({DBGAP_VAR_URL.format(study=study_accession, phv=int(num))})"


# ---------------------------------------------------------------------------
# Reusable list of variables with a +/- button per row
# ---------------------------------------------------------------------------

def variable_rows(df, key, show_study=True, extra_col=None):
    """Paginated list: [+/−] name | description | study / dataset | N | unit | variable id."""
    if df is None or df.empty:
        st.caption("No variables.")
        return
    total = len(df)
    pages = (total - 1) // PAGE_SIZE + 1
    top = st.columns([3, 2, 2])
    not_in = [u for u in df["uri"] if u not in st.session_state.cohort]
    top[0].caption(f"{total:,} variable(s)" + (f" · page size {PAGE_SIZE}" if pages > 1 else ""))
    page = top[1].number_input("Page", 1, pages, 1, key=f"{key}_page",
                               label_visibility="collapsed") if pages > 1 else 1
    top[2].button(f"➕ Add all {len(not_in):,} to cohort", key=f"{key}_addall",
                  on_click=cohort_add, args=tuple(not_in), disabled=not not_in)

    widths = [0.5, 1.6, 3.4, 1.6, 0.8, 0.9, 1.5]
    head = st.columns(widths)
    for col, h in zip(head, ["", "Variable", "Description",
                             "Study / dataset" if show_study else "Dataset", "N", "Unit", "Variable ID"]):
        col.markdown(f"**{h}**")

    start = (page - 1) * PAGE_SIZE
    for r in df.iloc[start:start + PAGE_SIZE].itertuples():
        cols = st.columns(widths)
        if r.uri in st.session_state.cohort:
            cols[0].button("−", key=f"{key}_rm_{r.uri}", on_click=cohort_remove, args=(r.uri,),
                           help="Remove from cohort")
        else:
            cols[0].button("+", key=f"{key}_add_{r.uri}", on_click=cohort_add, args=(r.uri,),
                           help="Add to cohort")
        cols[1].markdown(f"`{r.label}`")
        desc = r.comment or ""
        if extra_col and getattr(r, extra_col, None):
            desc += f"  \n*via value: {getattr(r, extra_col)}*"
        cols[2].markdown(desc)
        where = r.datasets or ""
        if show_study:
            where = f"{r.study_name or ''}" + (f" / {r.datasets}" if r.datasets else "")
        cols[3].markdown(where)
        cols[4].markdown(fmt_n(r.n))
        cols[5].markdown(r.unit or "")
        vid = getattr(r, "dataset_variable_id", None) or r.variable_id
        cols[6].markdown(phv_link(vid, r.study_accession))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
if not DB_PATH.exists():
    st.error(f"{DB_PATH.name} not found next to app.py. Build it with parse_owl.py first.")
    st.stop()

st.title("◈ BDC Variable Explorer")
st.caption("Browse BioData Catalyst (dbGaP) variables by study, dataset, unit, value and "
           "ontology term, and collect them into a cohort.")

counts = load_counts()
m = st.columns(6)
m[0].metric("Variables", f"{counts['variables']:,}")
m[1].metric("Studies", f"{counts['studies']:,}")
m[2].metric("Datasets", f"{counts['datasets']:,}")
m[3].metric("Units", f"{counts['units']:,}")
m[4].metric("Ontology terms", f"{counts['onto']:,}")
m[5].metric("In cohort", f"{len(st.session_state.cohort):,}")

st.divider()

tab_names = ["\U0001F50D Browse by Variable", "\U0001F4C1 Browse by Dataset", "\U0001F4CF Browse by Unit",
             "\U0001F3F7️ Browse by Value", ":material/account_tree: Browse by Ontology term"]
if counts["bundles"]:
    tab_names.append("\U0001F4CB Browse by Bundle")
if counts["cdm"]:
    tab_names.append("▦ Browse by CDM")
# Keep tab labels fixed: if a label changed (e.g. a live cohort count), Streamlit would
# jump back to the first tab every time + or − is clicked. The count is in the metrics row.
tab_names += ["\U0001F9FA Cohort", "ℹ️ About"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))
T = lambda prefix: next(t for name, t in tabs.items() if prefix in name)

# ---------------- Browse by Variable ----------------
with T("by Variable"):
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Filters")
        text = st.text_input("Search name / description / phv id", key="v_text")
        studies = load_studies()
        study_lbl = {"": ""} | {r.study_accession: f"{r.study_name} ({r.study_accession})"
                                for r in studies.itertuples()}
        study = st.selectbox("Study", list(study_lbl), format_func=study_lbl.get, key="v_study")
        ds = load_datasets(study or None)
        ds_lbl = {"": ""} | {r.dataset_id: f"{r.dataset_name} ({r.dataset_id})" for r in ds.itertuples()}
        dataset = st.selectbox("Dataset", list(ds_lbl), format_func=ds_lbl.get, key="v_dataset")
        unit = st.selectbox("Unit", [""] + load_units()["unit"].tolist(), key="v_unit")
        ontology = st.selectbox("Ontology", [""] + load_ontologies(), key="v_onto")

        results = search_variables(text or None, study or None, dataset or None, unit or None, ontology or None)
        st.caption(f"{len(results)} variable(s) match (showing up to 500)")
        not_in = [u for u in results["uri"] if u not in st.session_state.cohort]
        st.button(f"➕ Add all {len(not_in)} matches to cohort", key="v_addall",
                  on_click=cohort_add, args=tuple(not_in), disabled=not not_in)
        # Labels must not depend on the cohort: if a label changes between reruns (e.g. a ✓ that
        # disappears on "Clear cohort"), the browser's selection no longer matches any option.
        labels = {r.uri: f"{r.label}  ·  {r.study_name}" for r in results.itertuples()}
        selected = st.radio("Select a variable", results["uri"].tolist(), format_func=lambda u: labels.get(u, u),
                            key="v_pick") if len(results) else None

    with right:
        st.subheader("Variable detail")
        detail = get_variable_detail(selected) if selected else None
        if detail is None or detail[0].empty:
            st.info("No variable selected. Adjust the filters on the left, or pick one from the list.")
        else:
            cde, datasets, values, onto, bundles, cdm = detail
            row = cde.iloc[0]
            h1, h2 = st.columns([5, 1])
            h1.markdown(f"### {row['label']}")
            if selected in st.session_state.cohort:
                h2.button("− Remove from cohort", key="d_rm", on_click=cohort_remove, args=(selected,))
            else:
                h2.button("+ Add to cohort", key="d_add", on_click=cohort_add, args=(selected,), type="primary")
            if row["comment"]:
                st.markdown(f"> {row['comment']}")

            study_link = (f"[{row['study_accession']}]({DBGAP_STUDY_URL.format(study=row['study_accession'])})"
                          if row["study_accession"] else "")
            facts = {
                "Study": f"{row['study_name'] or ''} {study_link}",
                "N": fmt_n(row["n"]),
                "Unit": row["unit"] or "—",
                "dbGaP type": row["data_type"] or "—",
            }
            st.markdown("  \n".join(f"**{k}:** {v}" for k, v in facts.items()))
            if row["value_counts"]:
                st.markdown(f"**Most frequent values:** {row['value_counts']}")

            dt, vt, ot = st.tabs([f"Datasets ({len(datasets)})", f"Values ({len(values)})",
                                  f"Ontology terms ({len(onto)})"])
            with dt:
                for r in datasets.itertuples():
                    st.markdown(f"- **{r.dataset_name}** (`{r.dataset_id}`"
                                + (f" v{int(r.dataset_version)}" if pd.notna(r.dataset_version) else "")
                                + f") — {phv_link(r.variable_id, r.study_accession)}")
                if datasets.empty:
                    st.caption("No dataset information.")
            with vt:
                if len(values):
                    st.write(", ".join(f"`{v}`" for v in values["value_label"]))
                else:
                    st.caption("No enumerated values.")
            with ot:
                for r in onto.itertuples():
                    links = " · ".join(f"[{name}]({url})" for name, url in
                                       (("OLS", r.ols_url), ("BioPortal", r.bioportal_url)) if url)
                    via = f"  *(via value `{r.via_value}`)*" if r.via_value else ""
                    st.markdown(f"- `{r.term_id}` {r.label or ''} — {r.ontology} — {links}{via}")
                if onto.empty:
                    st.caption("No ontology terms linked.")
            if len(bundles):
                st.markdown("**Bundles**")
                st.dataframe(bundles, hide_index=True)
            if len(cdm):
                st.markdown("**CDM mappings**")
                st.dataframe(cdm, hide_index=True)

# ---------------- Browse by Dataset ----------------
with T("by Dataset"):
    studies = load_studies()
    if studies.empty:
        st.info("No study information in the database.")
    else:
        c1, c2 = st.columns(2)
        s_lbl = {r.study_accession: f"{r.study_name} ({r.study_accession}) — {r.n_datasets} datasets, "
                                    f"{r.n_vars:,} variables" for r in studies.itertuples()}
        s_pick = c1.selectbox("Study", list(s_lbl), format_func=s_lbl.get, key="ds_study")
        dsets = load_datasets(s_pick)
        d_lbl = {r.dataset_id: f"{r.dataset_name} ({r.dataset_id}) — {r.n_vars:,} variables"
                 for r in dsets.itertuples()}
        d_pick = c2.selectbox("Dataset", list(d_lbl), format_func=d_lbl.get, key="ds_pick") if d_lbl else None
        if d_pick:
            df = variables_in_dataset(d_pick)
            f = st.text_input("Filter variables in this dataset", key="ds_filter")
            if f:
                df = df[df["label"].str.contains(f, case=False, na=False)
                        | df["comment"].str.contains(f, case=False, na=False)]
            variable_rows(df, key=f"ds_{d_pick}", show_study=False)

# ---------------- Browse by Unit ----------------
with T("by Unit"):
    units = load_units()
    if units.empty:
        st.info("No units in the database.")
    else:
        u_search = st.text_input("Search units", key="u_search")
        uf = units[units["unit"].str.contains(u_search, case=False, na=False)] if u_search else units
        u_n = dict(zip(uf["unit"], uf["n_vars"]))
        u_pick = st.selectbox("Unit", uf["unit"].tolist(), format_func=lambda u: f"{u}  ({u_n.get(u, 0):,} variables)",
                              key="u_pick") if len(uf) else None
        if u_pick:
            variable_rows(variables_with_unit(u_pick), key=f"u_{u_pick}")

# ---------------- Browse by Value ----------------
with T("by Value"):
    vdf = load_value_options()
    v_search = st.text_input("Search value text", key="val_search")
    vf = vdf[vdf["value_label"].str.contains(v_search, case=False, na=False)] if v_search else vdf
    st.caption(f"{len(vf):,} distinct value(s) (of {len(vdf):,})")
    v_n = dict(zip(vf["value_label"], vf["n"]))
    v_pick = st.selectbox("Value", vf["value_label"].tolist(),
                          format_func=lambda v: f"{v}  ({v_n.get(v, '?')} variables)", key="val_pick") if len(vf) else None
    if v_pick:
        variable_rows(variables_with_value(v_pick), key=f"val_{v_pick}")

# ---------------- Browse by Ontology term ----------------
with T("by Ontology"):
    odf = load_onto_df()
    o1, o2 = st.columns([1, 2])
    o_ont = o1.selectbox("Ontology", ["All"] + sorted(odf["ontology"].unique().tolist()), key="o_ont")
    o_search = o2.text_input("Search term name or ID", key="o_search")
    of = odf if o_ont == "All" else odf[odf["ontology"] == o_ont]
    if o_search:
        of = of[of["term_id"].str.contains(o_search, case=False, na=False)
                | of["label"].fillna("").str.contains(o_search, case=False, na=False)]
    st.caption(f"{len(of):,} term(s) linked to at least one variable")
    o_lbl = {r.uri: f"{r.term_id}" + (f" — {r.label}" if r.label else "") + f"  ({r.n_vars} variables)"
             for r in of.itertuples()}
    o_pick = st.selectbox("Term", list(o_lbl), format_func=o_lbl.get, key="o_pick") if o_lbl else None
    if o_pick:
        t = odf[odf.uri == o_pick].iloc[0]
        links = " · ".join(f"[Open in {name}]({url})" for name, url in
                           (("OLS", t["ols_url"]), ("BioPortal", t["bioportal_url"])) if url)
        st.markdown(f"### {t['term_id']}" + (f" — {t['label']}" if t["label"] else "") + f"\n{links}")
        variable_rows(variables_for_term(o_pick), key=f"o_{o_pick}", extra_col="via_value")

# ---------------- Bundle / CDM (only shown when the DB has them) ----------------
if counts["bundles"]:
    with T("by Bundle"):
        b = q("SELECT uri, label, kind FROM bundle ORDER BY label")
        b_lbl = dict(zip(b["uri"], b["label"]))
        b_pick = st.selectbox("Bundle", list(b_lbl), format_func=b_lbl.get, key="b_pick")
        if b_pick:
            variable_rows(q(f"SELECT {VAR_COLS} FROM cde_bundle cb JOIN cde c ON c.uri = cb.cde_uri "
                            "WHERE cb.bundle_uri = ? ORDER BY c.label", [b_pick]), key=f"b_{b_pick}")
if counts["cdm"]:
    with T("by CDM"):
        cm = q("SELECT uri, table_name, column_name FROM cdm ORDER BY table_name, column_name")
        cm_lbl = {r.uri: f"{r.table_name}.{r.column_name}" for r in cm.itertuples()}
        cm_pick = st.selectbox("CDM column", list(cm_lbl), format_func=cm_lbl.get, key="cm_pick")
        if cm_pick:
            variable_rows(q(f"SELECT {VAR_COLS} FROM cde_cdm cc JOIN cde c ON c.uri = cc.cde_uri "
                            "WHERE cc.cdm_uri = ? ORDER BY c.label", [cm_pick]), key=f"cm_{cm_pick}")

# ---------------- Cohort ----------------
with T("Cohort"):
    uris = tuple(st.session_state.cohort)
    if not uris:
        st.info("Your cohort is empty. Use **+** next to any variable in the other tabs to add it here.")
    else:
        rows = cohort_rows(uris)
        k = st.columns([1, 1, 1, 2, 2])
        k[0].metric("Variables", f"{len(uris):,}")
        k[1].metric("Studies", f"{rows['study_accession'].nunique():,}")
        k[2].metric("Datasets", f"{rows['dataset_id'].nunique():,}")
        export = rows[["study_name", "study_accession", "dataset_name", "dataset_id",
                       "label", "variable_id", "n", "unit", "comment"]].rename(
            columns={"label": "variable", "comment": "description"})
        k[3].download_button("⬇️ Download cohort (CSV)", export.to_csv(index=False),
                             file_name="bdc_cohort.csv", mime="text/csv")
        k[4].button("\U0001F5D1️ Clear cohort", on_click=cohort_clear, key="c_clear")
        st.caption("**Max possible N** is an upper bound: participants with *all* of the selected "
                   "variables can't exceed the smallest N among them. The actual overlap needs "
                   "participant-level data. Variables without an N (e.g. IDs) are left out of the bound. "
                   "A variable that occurs in more than one dataset is listed under each.")

        def bound(df):
            ns = df["n"].dropna()
            return f"{int(ns.min()):,}" if len(ns) else "—"

        for (sname, sacc), sdf in rows.groupby(["study_name", "study_accession"], sort=False, dropna=False):
            n_vars = sdf["uri"].nunique()
            with st.expander(f"**{sname}** ({sacc}) — {n_vars} variable(s) · max possible N ≤ {bound(sdf)}",
                             expanded=True):
                for (dname, did), ddf in sdf.groupby(["dataset_name", "dataset_id"], sort=False, dropna=False):
                    did_txt = f" (`{did}`)" if isinstance(did, str) else ""
                    head = st.columns([5, 1.2, 1.3])
                    head[0].markdown(f"\U0001F4C1 **{dname}**{did_txt} — {len(ddf)} variable(s) · "
                                     f"max possible N ≤ {bound(ddf)}")
                    head[2].button("− Remove dataset", key=f"c_rmds_{sacc}_{did}",
                                   on_click=cohort_remove, args=tuple(ddf["uri"].unique()),
                                   help="Remove all of this dataset's variables from the cohort")
                    # Draw at most COHORT_PAGE_SIZE rows per dataset at a time: every click
                    # redraws every tab, so a very large cohort would otherwise slow the whole app.
                    pages = (len(ddf) - 1) // COHORT_PAGE_SIZE + 1
                    page = head[1].number_input("Page", 1, pages, 1, key=f"c_page_{sacc}_{did}",
                                                label_visibility="collapsed",
                                                help=f"{pages} pages") if pages > 1 else 1
                    widths = [0.4, 1.8, 1.6, 0.9, 0.9, 3.4]
                    for col, h in zip(st.columns(widths), ["", "Variable", "Variable ID", "N", "Unit", "Description"]):
                        col.caption(h)
                    start = (page - 1) * COHORT_PAGE_SIZE
                    for r in ddf.iloc[start:start + COHORT_PAGE_SIZE].itertuples():
                        c = st.columns(widths)
                        c[0].button("−", key=f"c_rm_{r.uri}_{did}", on_click=cohort_remove, args=(r.uri,),
                                    help="Remove from cohort")
                        c[1].markdown(f"`{r.label}`")
                        c[2].markdown(phv_link(r.variable_id, r.study_accession))
                        c[3].markdown(fmt_n(r.n))
                        c[4].markdown(r.unit or "")
                        c[5].markdown(r.comment or "")

# ---------------- About ----------------
# name -> (link, shown ✅ when this acronym appears in onto_term.ontology)
ONTOLOGY_REFS = [
    ("Human Phenotype Ontology (HPO)", "https://hpo.jax.org/", ("HPO", "HP")),
    ("MONDO Disease Ontology", "https://mondo.monarchinitiative.org/", ("MONDO",)),
    ("LOINC", "https://loinc.org/", ("LOINC",)),
    ("UBERON (anatomy)", "https://obophenotype.github.io/uberon/", ("UBERON",)),
    ("CHEBI (chemicals)", "https://www.ebi.ac.uk/chebi/", ("CHEBI",)),
    ("Gene Ontology (GO)", "https://geneontology.org/", ("GO",)),
    ("VALUESETS", "https://bioportal.bioontology.org/ontologies/VALUESETS", ("VALUESETS",)),
]
ARCHITECTURE_CREDIT = ""   # e.g. "Diagram generated with the help of [Tool](https://...)."
BDC_STUDIES_URL = "https://biodatacatalyst.nhlbi.nih.gov/about/studies"
LEOS_URL = "https://biodatacatalyst.nhlbi.nih.gov/about/studies/leos"

with T("About"):
    logo_files = [p for p in (TISLAB_LOGO, UNC_LOGO) if p.exists()]
    if logo_files:
        for col, logo_path in zip(st.columns(len(logo_files)), logo_files):
            col.image(str(logo_path), width=180)

    st.markdown("### How this data is organized")
    st.markdown(
        "Every **variable** comes from a **dbGaP study** in BioData Catalyst and sits in one or more "
        "**datasets** (dbGaP `pht` tables). Each variable carries its own metadata (dbGaP variable id, "
        "**N**, unit, type and answer **values**) and is linked to terms in external **ontologies**, "
        "either directly or through one of its values. Variables you pick with **+** form a **cohort**."
    )

    if ARCHITECTURE_IMG.exists():
        with st.container(border=True):
            st.image(str(ARCHITECTURE_IMG), width=1350)   # fixed size; capped to screen width on small screens
        st.caption("\U0001F50D Hover over the top-right corner of the diagram and click the expand icon "
                   "to view it fullscreen.")
        if ARCHITECTURE_CREDIT:
            st.caption(ARCHITECTURE_CREDIT)

    st.markdown("### Studies in this proof of concept")
    st.markdown(
        f"[BioData Catalyst]({BDC_STUDIES_URL}) hosts many NHLBI studies. This POC covers part of "
        f"the [LEOS]({LEOS_URL}) studies, listed below with their dbGaP study pages."
    )
    studies_tbl = load_studies().assign(
        dbGaP=lambda d: [DBGAP_STUDY_URL.format(study=s) for s in d["study_accession"]])
    st.dataframe(
        studies_tbl[["study_name", "study_accession", "n_datasets", "n_vars", "dbGaP"]],
        hide_index=True,
        column_config={
            "study_name": "Study", "study_accession": "Accession",
            "n_datasets": "Datasets", "n_vars": "Variables",
            "dbGaP": st.column_config.LinkColumn("dbGaP", display_text="Open"),
        },
    )

    st.markdown("### Key technical & design highlights")
    st.markdown("**Pipeline:** dbGaP `var_report.xml` → LinkML → ontology annotation → SQLite → this app")
    onts = ", ".join(load_ontologies()) or "—"
    st.markdown(f"""
1. **LinkML as the schema backbone** — dbGaP variable reports converted to LinkML: one class per
   dataset (`pht` table) and one slot per variable (`phv`), with enums for categorical answers.
   The schema mirrors dbGaP's study → dataset → variable structure (identical variables that
   share a name are merged into one slot that keeps all their `phv` ids), and LinkML can generate
   JSON Schema or Python data classes from it.
2. **Two-level ontology mapping**:
   - *Variable level* — variable names and descriptions matched to ontology terms.
   - *Value level* — categorical answers matched to terms of their own, e.g. an answer like
     `Never` matched to a VALUESETS term.

   Ontologies in use: {onts}.
3. **Indexed SQLite for fast browsing** — the annotated variables are stored in SQLite with indexes
   on the lookups each tab makes, so browsing by variable, dataset, unit, value or ontology term
   stays fast across all studies.
4. **Session cohort basket** — collect variables across studies and datasets, see each one's **N** and
   an upper bound on the cohort size (the smallest **N** among them), and export the selection as CSV.
""")

    st.markdown("### How to explore with the tabs above")
    st.markdown("""
- **\U0001F50D Browse by Variable** — search by name, description or `phv` id and filter by study, dataset,
  unit or ontology. Open a variable to see its study, N, unit, type, most frequent values, datasets,
  values and ontology terms (with links to OLS and BioPortal).
- **\U0001F4C1 Browse by Dataset** — pick a study, then one of its datasets, to list its variables.
- **\U0001F4CF Browse by Unit** — every variable recorded in a unit, e.g. `mmHg` or `kg/m2`.
- **\U0001F3F7️ Browse by Value** — every variable that uses an answer value, e.g. `Never`.
- **:material/account_tree: Browse by Ontology term** — every variable linked to a term.
- **\U0001F9FA Cohort** — the variables you added with **+**, grouped study → dataset, with N and an
  upper bound on the possible N. Remove with **−** or download as CSV. The cohort lasts for your
  browser session only, so download it to keep it.
""")

    st.markdown("### Reference links")
    st.caption("✅ marks the sources this app draws data from today; the rest are shown for context.")
    used = {o.upper() for o in load_ontologies()}
    tick = lambda name, url, on: f"- ✅ :green[[{name}]({url})]" if on else f"- [{name}]({url})"
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Data sources**\n" + "\n".join([
            tick("NHLBI BioData Catalyst", "https://biodatacatalyst.nhlbi.nih.gov/", True),
            tick("BDC studies", BDC_STUDIES_URL, True),
            tick("LEOS studies in BDC", LEOS_URL, True),
            tick("dbGaP", "https://www.ncbi.nlm.nih.gov/gap/", True),
        ]))
        st.markdown("**Ontologies**\n" + "\n".join(
            tick(name, url, any(a in used for a in acrs)) for name, url, acrs in ONTOLOGY_REFS))
    with r2:
        st.markdown("**Term lookup services**\n" + "\n".join([
            tick("OLS — Ontology Lookup Service (EMBL-EBI)", "https://www.ebi.ac.uk/ols4/", True),
            tick("BioPortal", "https://bioportal.bioontology.org/", True),
        ]))
        st.markdown("**Standards**\n" + "\n".join([
            tick("LinkML", "https://linkml.io/", True),
        ]))

    st.divider()
    st.warning(
        "\U0001F6A7 **Disclaimer:** This is a proof-of-concept (POC) for exploring BioData Catalyst "
        "variables, their datasets, values and ontology mappings. Data comes from publicly available "
        "dbGaP variable reports. Ontology mappings are automatic text matches and may be wrong. "
        "**N** is dbGaP's record count for each variable, not a participant-level cohort count."
    )