"""
Query Language Engine — Cypher-subset + SQL compatibility layer.

Supports:
  MATCH ... WHERE ... RETURN  (graph pattern matching)
  EXPLAIN / ANALYZE           (query plan inspection)
  SQL SELECT ... FROM nodes WHERE ...
  Parameterised queries       (prevents injection)
"""
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nextgendb.query.lang")


# ── Query Plan Node ───────────────────────────────────────────────────────────

@dataclass
class PlanNode:
    op:       str
    details:  str
    cost_est: float = 0.0
    children: List["PlanNode"] = field(default_factory=list)

    def explain(self, indent: int = 0) -> str:
        lines = [" " * indent + f"[{self.op}] {self.details}  (est_cost={self.cost_est:.2f})"]
        for child in self.children:
            lines.append(child.explain(indent + 2))
        return "\n".join(lines)


@dataclass
class QueryResult:
    rows:        List[Dict[str, Any]]
    plan:        Optional[PlanNode]
    latency_ms:  float
    scanned:     int
    returned:    int


# ── Cypher Tokeniser (Minimal) ─────────────────────────────────────────────────

_CYPHER_MATCH     = re.compile(r"MATCH\s+(.+?)(?:\s+WHERE\s+(.+?))?\s+RETURN\s+(.+)", re.IGNORECASE | re.DOTALL)
_CYPHER_NODE_PAT  = re.compile(r"\((\w+)(?::(\w+))?(?:\s*\{(.+?)\})?\)")
_CYPHER_EDGE_PAT  = re.compile(r"-\[(\w+)?(?::(\w+))?(?:\*(\d+)?(?:\.\.(\d+))?)?\]->")
_PARAM_RE         = re.compile(r"\$(\w+)")

# ── SQL tokeniser (minimal) ───────────────────────────────────────────────────
_SQL_SELECT  = re.compile(
    r"SELECT\s+(.+?)\s+FROM\s+(\w+)"
    r"(?:\s+(?:INNER\s+|LEFT\s+|RIGHT\s+)?JOIN\s+(\w+)\s+ON\s+(.+?))?"
    r"(?:\s+WHERE\s+(.+?))?"
    r"(?:\s+GROUP\s+BY\s+(.+?))?"
    r"(?:\s+ORDER\s+BY\s+(.+?))?"
    r"(?:\s+LIMIT\s+(\d+))?",
    re.IGNORECASE | re.DOTALL,
)


class QueryParser:
    """Parse Cypher / SQL strings into structured query objects."""

    @staticmethod
    def parse_params(query: str, params: Dict[str, Any]) -> str:
        """Substitute $param placeholders safely."""
        def replace(m):
            key = m.group(1)
            if key not in params:
                raise ValueError(f"Missing query parameter: ${key}")
            val = params[key]
            return f'"{val}"' if isinstance(val, str) else str(val)
        return _PARAM_RE.sub(replace, query)

    @classmethod
    def parse(cls, query: str, params: Optional[Dict[str, Any]] = None) -> dict:
        if params:
            query = cls.parse_params(query, params)
        q = query.strip()
        if q.upper().startswith("MATCH") or q.upper().startswith("OPTIONAL"):
            return {"type": "CYPHER", "raw": q, **cls._parse_cypher(q)}
        if q.upper().startswith("SELECT"):
            return {"type": "SQL", "raw": q, **cls._parse_sql(q)}
        if q.upper().startswith("EXPLAIN"):
            inner = q[7:].strip()
            parsed = cls.parse(inner)
            return {"type": "EXPLAIN", "inner": parsed}
        if q.upper().startswith("ANALYZE") or q.upper().startswith("ANALYSE"):
            inner = re.split(r"\s+", q, 1)[1].strip()
            parsed = cls.parse(inner)
            return {"type": "ANALYZE", "inner": parsed}
        # Fallback: treat as natural language (pass to agent)
        return {"type": "NATURAL", "raw": q}

    @staticmethod
    def _parse_cypher(q: str) -> dict:
        m = _CYPHER_MATCH.match(q)
        if not m:
            return {"error": "Cannot parse MATCH clause"}
        pattern_str, where_str, return_str = m.group(1), m.group(2), m.group(3)

        nodes = [{"alias": n[0], "label": n[1], "props": n[2]} for n in _CYPHER_NODE_PAT.findall(pattern_str)]
        
        edges = []
        for e in _CYPHER_EDGE_PAT.findall(pattern_str):
            alias, typ, min_hops, max_hops = e
            hops = None
            if min_hops or max_hops:
                # E.g., [*1..3] -> min_hops=1, max_hops=3
                hops = (int(min_hops) if min_hops else 1, int(max_hops) if max_hops else None)
            edges.append({"alias": alias, "type": typ, "hops": hops})
        returns = [r.strip() for r in return_str.split(",")]

        # Parse simple WHERE conditions: alias.prop OPERATOR value
        filters = []
        if where_str:
            for cond in re.split(r"\s+AND\s+", where_str, flags=re.IGNORECASE):
                fm = re.match(r"(\w+)\.(\w+)\s*(=|<|>|<=|>=|<>|CONTAINS|STARTS\s+WITH)\s*(.+)", cond.strip(), re.IGNORECASE)
                if fm:
                    filters.append({"alias": fm.group(1), "prop": fm.group(2), "op": fm.group(3).strip(), "val": fm.group(4).strip().strip('"\'')})

        return {"nodes": nodes, "edges": edges, "filters": filters, "returns": returns}

    @staticmethod
    def _parse_sql(q: str) -> dict:
        m = _SQL_SELECT.match(q)
        if not m:
            return {"error": "Cannot parse SELECT"}
        cols_str, table, join_table, join_on_str, where_str, group_str, order_str, limit_str = m.groups()
        columns = [c.strip() for c in cols_str.split(",")]
        limit   = int(limit_str) if limit_str else None
        
        joins = []
        if join_table and join_on_str:
            joins.append({"table": join_table, "on": join_on_str.strip()})
            
        filters = []
        if where_str:
            for cond in re.split(r"\s+AND\s+", where_str, flags=re.IGNORECASE):
                fm = re.match(r"([\w\.]+)\s*(=|<|>|<=|>=|<>|LIKE)\s*(.+)", cond.strip(), re.IGNORECASE)
                if fm:
                    filters.append({"col": fm.group(1).strip(), "op": fm.group(2).strip(), "val": fm.group(3).strip().strip('"\'')})
                    
        group_by = [g.strip() for g in group_str.split(",")] if group_str else []
        
        order_by = []
        if order_str:
            for o in order_str.split(","):
                parts = o.strip().split()
                col = parts[0]
                direction = parts[1].upper() if len(parts) > 1 else "ASC"
                order_by.append({"col": col, "dir": direction})
                
        return {
            "table": table,
            "columns": columns,
            "joins": joins,
            "filters": filters,
            "group_by": group_by,
            "order_by": order_by,
            "limit": limit
        }


# ── Query Executor ─────────────────────────────────────────────────────────────

class QueryExecutor:
    """Execute parsed Cypher/SQL queries against the PersistentGraphEngine."""

    def __init__(self, engine):
        self.engine = engine

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None, analyze: bool = False) -> QueryResult:
        t0      = time.perf_counter()
        parsed  = QueryParser.parse(query, params)

        plan    = self._build_plan(parsed)
        explain = parsed["type"] in ("EXPLAIN",)

        if explain:
            parsed = parsed["inner"]
            plan   = self._build_plan(parsed)

        if analyze:
            rows, scanned = self._run(parsed)
            latency_ms = (time.perf_counter() - t0) * 1000
            return QueryResult(rows=rows, plan=plan, latency_ms=latency_ms, scanned=scanned, returned=len(rows))

        if explain:
            return QueryResult(rows=[], plan=plan, latency_ms=0.0, scanned=0, returned=0)

        rows, scanned = self._run(parsed)
        latency_ms = (time.perf_counter() - t0) * 1000
        return QueryResult(rows=rows, plan=plan, latency_ms=latency_ms, scanned=scanned, returned=len(rows))

    def _build_plan(self, parsed: dict) -> PlanNode:
        qtype = parsed.get("type", "UNKNOWN")
        if qtype == "CYPHER":
            filter_plan = PlanNode("Filter", f"WHERE {len(parsed.get('filters', []))} conditions", cost_est=0.5)
            scan_plan   = PlanNode("NodeScan", f"labels={[n.get('label') for n in parsed.get('nodes', [])]}", cost_est=1.0 + self.engine.graph.number_of_nodes() * 0.001, children=[filter_plan])
            
            if parsed.get("edges"):
                scan_plan = PlanNode("EdgeScan", f"hops={parsed['edges'][0].get('hops')}", cost_est=scan_plan.cost_est + self.engine.graph.number_of_edges() * 0.002, children=[scan_plan])
                
            return PlanNode("CypherMatch", "MATCH pattern traversal", cost_est=scan_plan.cost_est + 0.1, children=[scan_plan])
            
        if qtype == "SQL":
            table = parsed.get("table", "")
            scan  = PlanNode("TableScan", f"FROM {table}", cost_est=self.engine.graph.number_of_nodes() * 0.002)
            
            plan = scan
            joins = parsed.get("joins", [])
            if joins:
                for j in joins:
                    plan = PlanNode("NestedLoopJoin", f"JOIN {j['table']}", cost_est=plan.cost_est * self.engine.graph.number_of_nodes() * 0.001, children=[plan])
            
            if parsed.get("filters"):
                plan = PlanNode("Filter", f"WHERE {len(parsed['filters'])} conditions", cost_est=plan.cost_est + 0.5, children=[plan])
                
            if parsed.get("group_by"):
                plan = PlanNode("HashAggregate", f"GROUP BY {parsed['group_by']}", cost_est=plan.cost_est + 1.0, children=[plan])
                
            if parsed.get("order_by"):
                plan = PlanNode("Sort", f"ORDER BY {parsed['order_by']}", cost_est=plan.cost_est + 2.0, children=[plan])

            return PlanNode("SQLSelect", f"SELECT {parsed.get('columns', ['*'])}", cost_est=plan.cost_est + 0.05, children=[plan])
            
        return PlanNode("NLQuery", "Natural language → agent pipeline", cost_est=5.0)

    def _run(self, parsed: dict) -> Tuple[List[Dict], int]:
        qtype = parsed.get("type", "UNKNOWN")
        if qtype == "CYPHER":
            return self._run_cypher(parsed)
        if qtype == "SQL":
            return self._run_sql(parsed)
        # Natural language — return empty; caller should route to agent
        return [], 0

    def _run_cypher(self, parsed: dict) -> Tuple[List[Dict], int]:
        g = self.engine.graph
        scanned = 0
        candidates: List[str] = []

        # Find nodes matching first pattern node label/props
        target_label = None
        target_props: Dict = {}
        first_alias = "n"
        if parsed.get("nodes"):
            n = parsed["nodes"][0]
            first_alias = n.get("alias") or "n"
            target_label = n.get("label")
            if n.get("props"):
                # parse inline props: "name: 'X', age: 30"
                for kv in re.split(r",\s*", n["props"]):
                    km = re.match(r"(\w+)\s*:\s*(.+)", kv.strip())
                    if km:
                        target_props[km.group(1)] = km.group(2).strip().strip("\"'")

        for node, data in g.nodes(data=True):
            scanned += 1
            if target_label and data.get("type", "").upper() != target_label.upper() and data.get("label", "").upper() != target_label.upper():
                continue
            match = True
            for k, v in target_props.items():
                if str(data.get(k, "")) != str(v):
                    match = False
                    break
            if match:
                candidates.append(node)

        paths = []
        if parsed.get("edges"):
            edge_spec = parsed["edges"][0]
            edge_alias = edge_spec.get("alias", "e")
            edge_type = edge_spec.get("type")
            hops = edge_spec.get("hops")
            min_hops = hops[0] if hops else 1
            max_hops = hops[1] if hops else 1
            
            second_node_spec = parsed["nodes"][1] if len(parsed.get("nodes", [])) > 1 else None
            second_alias = second_node_spec.get("alias", "m") if second_node_spec else "m"
            
            for start_node in candidates:
                queue = [(start_node, 0, [])]
                while queue:
                    curr, depth, path = queue.pop(0)
                    if min_hops <= depth <= max_hops:
                        paths.append({
                            first_alias: dict(g.nodes[start_node], _id=start_node),
                            second_alias: dict(g.nodes[curr], _id=curr),
                        })
                    if depth < max_hops:
                        for _, nbr, key, edata in g.out_edges(curr, keys=True, data=True):
                            scanned += 1
                            if edge_type and key != edge_type:
                                continue
                            queue.append((nbr, depth + 1, path + [(curr, key, nbr)]))
        else:
            for n in candidates:
                paths.append({first_alias: dict(g.nodes[n], _id=n)})

        # Apply WHERE filters
        rows: List[Dict] = []
        for path_data in paths:
            passed  = True
            for filt in parsed.get("filters", []):
                alias_data = path_data.get(filt["alias"], {})
                prop_val   = str(alias_data.get(filt["prop"], ""))
                filt_val   = str(filt["val"])
                op         = filt["op"].upper()
                if op == "=" and prop_val != filt_val:
                    passed = False
                elif op == "CONTAINS" and filt_val not in prop_val:
                    passed = False
                elif op == "STARTS WITH" and not prop_val.startswith(filt_val):
                    passed = False
            if passed:
                rows.append(path_data)

        # Project RETURN columns
        returns = parsed.get("returns", ["*"])
        if "*" not in returns and returns:
            projected = []
            for row in rows:
                proj = {}
                for ret in returns:
                    if "." in ret:
                        alias, prop = ret.split(".", 1)
                        proj[ret] = row.get(alias, {}).get(prop)
                    else:
                        proj[ret] = row.get(first_alias, {}).get(ret)
                projected.append(proj)
            rows = projected
        else:
            # Flatten to backward compatible structure if it was a single node match
            if not parsed.get("edges"):
                rows = [r.get(first_alias, {}) for r in rows]

        return rows, scanned

    def _run_sql(self, parsed: dict) -> Tuple[List[Dict], int]:
        table   = parsed.get("table", "nodes").lower()
        columns = parsed.get("columns", ["*"])
        joins   = parsed.get("joins", [])
        filters = parsed.get("filters", [])
        group_by = parsed.get("group_by", [])
        order_by = parsed.get("order_by", [])
        limit   = parsed.get("limit")
        g       = self.engine.graph
        scanned = 0

        def get_table_data(t):
            if t in ("nodes", "node"):
                return [{"_id": n, **dict(d)} for n, d in g.nodes(data=True)]
            elif t in ("edges", "edge", "relationships"):
                return [{"_src": s, "_tgt": t, "_type": k, **dict(d)} for s, t, k, d in g.edges(keys=True, data=True)]
            return []

        source = get_table_data(table)
        scanned += len(source)
        
        for join in joins:
            join_table = join["table"].lower()
            join_data = get_table_data(join_table)
            scanned += len(join_data)
            
            on_cond = join["on"]
            m = re.match(r"([\w\.]+)\s*=\s*([\w\.]+)", on_cond)
            if m:
                left_col = m.group(1).split(".")[-1]
                right_col = m.group(2).split(".")[-1]
                
                joined_source = []
                for left_row in source:
                    for right_row in join_data:
                        if str(left_row.get(left_col, "")) == str(right_row.get(right_col, "")):
                            merged = {**left_row, **right_row}
                            joined_source.append(merged)
                source = joined_source

        rows: List[Dict] = []
        for row in source:
            scanned += 1
            passed = True
            for filt in filters:
                col = filt["col"].split(".")[-1]
                val   = str(row.get(col, ""))
                fv    = str(filt["val"])
                op    = filt["op"].upper()
                if op == "="    and val != fv:    passed = False
                elif op == "<"  and not (val < fv):  passed = False
                elif op == ">"  and not (val > fv):  passed = False
                elif op == "LIKE" and fv.strip("%") not in val: passed = False
            if passed:
                rows.append(row)
                
        if group_by:
            grouped = {}
            for row in rows:
                key = tuple(row.get(g_col.split(".")[-1], "") for g_col in group_by)
                grouped.setdefault(key, []).append(row)
            rows = [group_rows[0] for group_rows in grouped.values()]
            
        if order_by:
            for ob in reversed(order_by):
                col = ob["col"].split(".")[-1]
                rev = ob["dir"] == "DESC"
                rows.sort(key=lambda r: str(r.get(col, "")), reverse=rev)

        final_rows = []
        for row in rows:
            if "*" in columns:
                final_rows.append(row)
            else:
                final_rows.append({c: row.get(c.split(".")[-1]) for c in columns})
                
        if limit and len(final_rows) > limit:
            final_rows = final_rows[:limit]

        return final_rows, scanned
