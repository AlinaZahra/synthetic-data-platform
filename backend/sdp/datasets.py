"""Fully synthetic demo datasets (no real personal data) used by tests, API and UI."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_customers(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Mixed-type table with correlations and a binary target (`churned`)."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(40, 12, n), 18, 85).round().astype(int)
    income = np.exp(10.4 + 0.02 * (age - 40) + rng.normal(0, 0.45, n)).round(2)
    tenure = np.clip(rng.exponential(24, n), 0, 120).round().astype(int)
    plan_score = np.log(income) + rng.normal(0, 0.5, n)
    plan = np.where(plan_score > 11.2, "enterprise", np.where(plan_score > 10.6, "pro", "basic"))
    region = rng.choice(["north", "south", "east", "west"], n, p=[0.4, 0.3, 0.2, 0.1])
    signup = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1800, n), unit="D")
    logit = -0.5 + 0.9 * (plan == "basic") - 0.03 * tenure + 0.01 * (50 - age) + rng.normal(0, 0.5, n)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "age": age, "income": income, "tenure_months": tenure, "plan": plan, "region": region,
        "signup_date": signup, "is_active": rng.random(n) < 0.8, "churned": churned,
    })


def make_shop(n_customers: int = 300, seed: int = 0) -> dict[str, pd.DataFrame]:
    """customers -> orders -> order_items <- products (all values invented)."""
    rng = np.random.default_rng(seed)
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "name": [f"Customer {i:04d}" for i in range(1, n_customers + 1)],
        "country": rng.choice(["US", "DE", "IN", "BR", "JP"], n_customers, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
        "segment": rng.choice(["retail", "smb", "corp"], n_customers, p=[0.6, 0.3, 0.1]),
    })
    n_products = 40
    products = pd.DataFrame({
        "product_id": np.arange(1, n_products + 1),
        "category": rng.choice(["tools", "toys", "books", "home"], n_products),
        "price": rng.gamma(4, 12, n_products).round(2),
    })
    per_customer = rng.negative_binomial(2, 2 / (2 + 4), n_customers)  # mean ~4, heavy tail, some zeros
    order_cust = np.repeat(customers["customer_id"].to_numpy(), per_customer)
    n_orders = len(order_cust)
    orders = pd.DataFrame({
        "order_id": np.arange(1, n_orders + 1),
        "customer_id": order_cust,
        "order_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n_orders), unit="D"),
        "status": rng.choice(["paid", "shipped", "returned"], n_orders, p=[0.3, 0.6, 0.1]),
    })
    per_order = 1 + rng.poisson(1.8, n_orders)
    item_order = np.repeat(orders["order_id"].to_numpy(), per_order)
    pop = rng.dirichlet(np.full(n_products, 0.6))
    prod_ids = rng.choice(products["product_id"], len(item_order), p=pop)
    items = pd.DataFrame({
        "item_id": np.arange(1, len(item_order) + 1),
        "order_id": item_order,
        "product_id": prod_ids,
        "quantity": 1 + rng.poisson(0.8, len(item_order)),
    })
    items["unit_price"] = products.set_index("product_id").loc[items["product_id"], "price"].to_numpy()
    return {"customers": customers, "products": products, "orders": orders, "order_items": items}


SHOP_FULL_RULES = [
    "orders.total = SUM(order_items.quantity * order_items.unit_price)",
    "order_items.unit_price = products.price",
    "shipments.shipped_date >= orders.order_date",
    "shipments.delivered_date >= shipments.shipped_date",
    "orders.discount <= 0.3",
    "orders.status = 'cancelled' => COUNT(shipments) = 0",
    "orders.status IN ('shipped', 'delivered') => COUNT(shipments) >= 1",
    "shipments.delivered_date IS NOT NULL => orders.status = 'delivered'",
]


def make_shop_full(n_customers: int = 400, seed: int = 0) -> dict[str, pd.DataFrame]:
    """customers -> orders -> order_items <- products, orders -> shipments.

    Built-in relations a good synthesizer must keep: segment drives order count, discount and basket size;
    orders.total = SUM(qty*unit_price); unit_price = products.price; shipments follow orders in time; status matches shipments.
    """
    rng = np.random.default_rng(seed)
    seg_p = {"retail": 0.6, "smb": 0.3, "corp": 0.1}
    seg = rng.choice(list(seg_p), n_customers, p=list(seg_p.values()))
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1), "name": [f"Customer {i:04d}" for i in range(1, n_customers + 1)],
        "country": rng.choice(["US", "DE", "IN", "BR", "JP"], n_customers), "segment": seg})
    n_products = 40
    products = pd.DataFrame({"product_id": np.arange(1, n_products + 1), "category": rng.choice(["tools", "toys", "books", "home"], n_products),
                             "price": np.round(rng.gamma(4, 12, n_products) + 1, 2)})
    mean_orders = {"retail": 3.0, "smb": 6.0, "corp": 12.0}
    per_cust = np.array([rng.poisson(mean_orders[s]) for s in seg])
    cust_of = np.repeat(customers["customer_id"].to_numpy(), per_cust)
    seg_of = np.repeat(seg, per_cust)
    n_orders = len(cust_of)
    order_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n_orders), unit="D")
    disc_choices = {"retail": ([0, 0.05], [0.9, 0.1]), "smb": ([0, 0.05, 0.1], [0.5, 0.3, 0.2]), "corp": ([0.1, 0.15, 0.2, 0.25], [0.3, 0.3, 0.25, 0.15])}
    discount = np.array([rng.choice(disc_choices[s][0], p=disc_choices[s][1]) for s in seg_of])
    status = rng.choice(["placed", "cancelled", "shipped", "delivered"], n_orders, p=[0.1, 0.08, 0.17, 0.65])
    orders = pd.DataFrame({"order_id": np.arange(1, n_orders + 1), "customer_id": cust_of, "order_date": order_date,
                           "status": status, "discount": discount})
    lam = {"retail": 1.0, "smb": 1.8, "corp": 3.0}
    n_items = 1 + np.array([rng.poisson(lam[s]) for s in seg_of])
    item_order = np.repeat(orders["order_id"].to_numpy(), n_items)
    item_seg = np.repeat(seg_of, n_items)
    pop = rng.dirichlet(np.full(n_products, 0.7))
    prod = rng.choice(products["product_id"], len(item_order), p=pop)
    qmean = {"retail": 0.6, "smb": 1.5, "corp": 4.0}
    qty = 1 + np.array([rng.poisson(qmean[s]) for s in item_seg])
    price = products.set_index("product_id").loc[prod, "price"].to_numpy()
    items = pd.DataFrame({"item_id": np.arange(1, len(item_order) + 1), "order_id": item_order, "product_id": prod,
                          "quantity": qty, "unit_price": price})
    orders["total"] = np.round(items.assign(v=items["quantity"] * items["unit_price"]).groupby("order_id")["v"].sum()
                               .reindex(orders["order_id"]).to_numpy(), 2)
    ship_rows = []
    sid = 1
    for oid, od, st in zip(orders["order_id"], orders["order_date"], orders["status"]):
        if st not in ("shipped", "delivered"):
            continue
        for _ in range(2 if (st == "delivered" and rng.random() < 0.15) else 1):
            shipped = od + pd.Timedelta(days=int(rng.integers(1, 5)))
            delivered = shipped + pd.Timedelta(days=int(rng.integers(1, 8))) if st == "delivered" else pd.NaT
            ship_rows.append((sid, oid, shipped, delivered, rng.choice(["DHL", "UPS", "FedEx"])))
            sid += 1
    shipments = pd.DataFrame(ship_rows, columns=["shipment_id", "order_id", "shipped_date", "delivered_date", "carrier"])
    return {"customers": customers, "products": products, "orders": orders, "order_items": items, "shipments": shipments}


def make_students(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Fictional university students with correlated study habits and a binary target (`passed`)."""
    rng = np.random.default_rng(seed)
    major = rng.choice(["engineering", "business", "arts", "science", "medicine"], n, p=[0.28, 0.24, 0.16, 0.22, 0.10])
    year = rng.integers(1, 5, n)
    age = np.clip(18 + (year - 1) + rng.normal(0.8, 1.6, n), 17, 34).round().astype(int)
    gpa = np.clip(rng.normal(3.0, 0.5, n) + 0.1 * (major == "medicine"), 1.0, 4.0).round(2)
    study = np.clip(rng.gamma(4.0, 3.0, n) + 4 * (gpa - 3.0), 0.5, 60).round(1)
    attendance = np.clip(rng.normal(82, 11, n) + 3 * (gpa - 3.0), 30, 100).round(1)
    job = rng.random(n) < 0.35
    scholarship = rng.random(n) < 1 / (1 + np.exp(-(3.0 * (gpa - 3.4))))
    enrolled = pd.Timestamp("2021-09-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D")
    score = np.clip(35 + 9 * gpa + 0.5 * study + 0.25 * attendance - 6 * job + rng.normal(0, 7, n), 0, 100).round(1)
    passed = (score + rng.normal(0, 4, n) > 84).astype(int)
    return pd.DataFrame({
        "age": age, "major": major, "year_of_study": year, "prior_gpa": gpa, "study_hours_week": study, "attendance_pct": attendance,
        "part_time_job": job, "scholarship": scholarship, "enrolled_date": enrolled, "final_score": score, "passed": passed,
    })


def make_employees(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Fictional employees: salary depends on department and tenure; the target is `left_company`."""
    rng = np.random.default_rng(seed)
    dept = rng.choice(["engineering", "sales", "support", "finance", "hr"], n, p=[0.32, 0.25, 0.2, 0.13, 0.10])
    age = np.clip(rng.normal(37, 9, n), 20, 66).round().astype(int)
    tenure = np.clip(rng.exponential(4.5, n), 0, 35).round(1)
    base = {"engineering": 78000, "sales": 62000, "support": 45000, "finance": 70000, "hr": 55000}
    salary = np.array([base[d] for d in dept]) * np.exp(0.03 * np.minimum(tenure, 15) + rng.normal(0, 0.15, n))
    rating = np.clip(np.round(rng.normal(3.3, 0.9, n)), 1, 5).astype(int)
    remote = rng.random(n) < np.where(dept == "engineering", 0.6, 0.25)
    hired = pd.Timestamp("2025-12-31") - pd.to_timedelta((tenure * 365).astype(int), unit="D")
    logit = -1.3 - 0.12 * tenure + 0.35 * (3 - rating) + 0.4 * (dept == "sales") - 0.000004 * (salary - 60000)
    left = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"age": age, "department": dept, "years_at_company": tenure, "salary": salary.round(2), "performance_rating": rating,
                         "works_remotely": remote, "hire_date": hired, "left_company": left})


# name -> (builder, target column for the utility test, display title)
TABULAR_SAMPLES = {
    "customers": (make_customers, "churned", "Customers (age, income, plan, churned)"),
    "students": (make_students, "passed", "Students (grades, study hours, passed)"),
    "employees": (make_employees, "left_company", "Employees (salary, department, left company)"),
}
