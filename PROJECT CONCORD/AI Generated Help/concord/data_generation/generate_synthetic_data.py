#!/usr/bin/env python3
"""
Project Concord -- synthetic data generator.

Produces one CSV per table, in dependency order, ready to be bulk-loaded
with `\\copy` (see load_csvs.sql in this same folder). Column order in
every CSV matches the column order in the migrations exactly.

WHY THIS SHAPE (see Appendix D of the brief):
  1. Generate the Core Hub identity pools FIRST (customers, employees,
     suppliers, locations, products) -- every division draws from these
     same pools rather than inventing its own disconnected identities.
  2. Build in the cross-divisional overlaps (farmers who are also VFS
     loan customers; loyalty customers who also hold a VFS wallet) at
     generation time, not bolted on afterward.
  3. Give transactional data (POS transactions, wallet transactions,
     harvest batches) a realistic time distribution instead of uniform
     randomness -- retail is busier on weekends, harvests cluster in a
     season.

SCALE: set the CONCORD_SCALE environment variable to generate a smaller
test dataset quickly, e.g.:
    CONCORD_SCALE=0.02 python3 generate_synthetic_data.py
Default (1.0) targets the LOW END of every range in brief section 5.4,
per the team's 4-week-timeline plan (see Team_Build_Plan.docx section 1).
Structural counts drawn directly from the brief's own company profile
(68 stores, 310 vehicles, 94 properties, 6 warehouses, 4 processing
facilities) are realistic dimension sizes, not "volumes" to be scaled
down -- they stay fixed regardless of SCALE, with only a small floor
applied so a tiny smoke-test run still produces a sane, loadable model.
"""
import os
import csv
import random
import numpy as np
from datetime import date, timedelta, datetime
from faker import Faker

# ============================================================================
# CONFIG
# ============================================================================
SCALE = float(os.environ.get("CONCORD_SCALE", "1.0"))
SEED = int(os.environ.get("CONCORD_SEED", "42"))

random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

OUT_DIR = os.environ.get("CONCORD_OUT_DIR", "output_csv")
os.makedirs(OUT_DIR, exist_ok=True)

def scaled(n_full, floor=1):
    return max(floor, int(round(n_full * SCALE)))

# --- Hub identity pool volumes (brief 5.4, low end of each range) ---------
N_CUSTOMERS = scaled(40_000, floor=50)
N_EMPLOYEES = scaled(1_500, floor=30)
N_SUPPLIERS = scaled(8_000, floor=40)          # includes farmers + other vendor types
N_PRODUCTS = scaled(400, floor=20)

# --- Structural / dimension counts (brief section 1.3, fixed real scale) --
N_STORES = scaled(68, floor=3)
N_WAREHOUSES = scaled(6, floor=1)
N_PROCESSING_FACILITIES = scaled(4, floor=1)
N_PROPERTIES = scaled(94, floor=5)              # "the full 94-property portfolio", 5.4
N_VEHICLES = scaled(310, floor=5)
N_ROUTES = scaled(40, floor=3)

# --- Transactional volumes (brief 5.4, low end) ---------------------------
N_POS_TRANSACTIONS = scaled(500_000, floor=500)
N_SHIPMENTS = scaled(20_000, floor=100)
N_WALLET_TRANSACTIONS = scaled(300_000, floor=300)
N_HARVEST_BATCHES = scaled(5_000, floor=50)
N_LEASES = scaled(300, floor=10)
N_MAINTENANCE_REQUESTS = scaled(300, floor=10)

# --- Cross-divisional overlap rates (brief 5.4 / Appendix D) --------------
PCT_FARMERS_WITH_VFS_LOAN = 0.28
PCT_LOYALTY_CUSTOMERS_WITH_WALLET = 0.50   # brief 1.3.1: "a large proportion"
PCT_SUPPLIERS_ARE_FARMERS = 0.70            # brief 5.4: "weighted heavily toward AgriCore"

COUNTRY_WEIGHTS = {"Nigeria": 0.70, "Ghana": 0.20, "Kenya": 0.10}
COUNTRY_CODE = {"Nigeria": "NG", "Ghana": "GH", "Kenya": "KE"}
CITY_BY_COUNTRY = {
    "Nigeria": ["Lagos", "Abuja", "Ibadan", "Kano", "Port Harcourt", "Benin City", "Enugu", "Uyo"],
    "Ghana": ["Accra", "Kumasi", "Tamale", "Takoradi"],
    "Kenya": ["Nairobi", "Mombasa", "Kisumu", "Nakuru"],
}
PHONE_PREFIX = {"Nigeria": "+234", "Ghana": "+233", "Kenya": "+254"}

DATE_START = date(2024, 1, 1)
DATE_END = date(2026, 6, 30)

def pick_country():
    countries = list(COUNTRY_WEIGHTS.keys())
    weights = list(COUNTRY_WEIGHTS.values())
    return random.choices(countries, weights=weights, k=1)[0]

def random_date(start=DATE_START, end=DATE_END):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def random_dates_vectorized(n, start=DATE_START, end=DATE_END):
    delta = (end - start).days
    offsets = np.random.randint(0, delta + 1, size=n)
    return [start + timedelta(days=int(o)) for o in offsets]

def phone_number(country):
    return f"{PHONE_PREFIX[country]}{random.randint(700000000, 909999999)}"

def write_csv(name, columns, rows):
    """rows: list of tuples/lists in column order."""
    path = os.path.join(OUT_DIR, f"{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"  wrote {len(rows):>9,} rows -> {name}.csv")

# ============================================================================
# 1. CORE SERVICES HUB -- identity pools everything else draws from
# ============================================================================

def generate_customers():
    print("Generating customers...")
    rows = []
    countries = []
    for cid in range(1, N_CUSTOMERS + 1):
        country = pick_country()
        countries.append(country)
        name = fake.name()
        dob = fake.date_of_birth(minimum_age=18, maximum_age=75)
        contact = phone_number(country)
        # Consent flags: most customers consent to retail personalisation
        # (it's the low-friction one); far fewer consent to their data
        # being visible to a financial-services credit assessment, since
        # that is the more sensitive cross-divisional use (brief 4.4).
        consent_retail = random.random() < 0.65
        consent_financial = random.random() < 0.30
        created = random_date(date(2019, 1, 1), DATE_END)
        rows.append((cid, name, dob.isoformat(), contact, country,
                     consent_retail, consent_financial, created.isoformat()))
    write_csv("customers", [
        "customer_id", "full_name", "date_of_birth", "primary_contact",
        "registered_country", "consent_retail_personalisation",
        "consent_cross_division_financial_view", "created_date"
    ], rows)
    return countries  # index 0 == customer_id 1, etc.

DIVISION_CODES = ["retail", "logistics", "vfs", "agricore", "properties", "group"]
# Employee headcount is a representative SAMPLE across divisions (brief
# 5.4: "not the group's full 14,600 headcount"), weighted roughly by each
# division's real relative size per section 1.3 (retail's 68 stores and
# logistics' driver/dispatch headcount are the biggest; VFS keeps a thin
# liaison headcount since its own banking staff sit outside this platform).
DIVISION_EMPLOYEE_WEIGHTS = {"retail": 0.40, "logistics": 0.20, "agricore": 0.20,
                             "properties": 0.10, "vfs": 0.05, "group": 0.05}
ROLE_TITLES_BY_DIVISION = {
    "retail": ["Store Manager", "Cashier", "Stock Clerk", "Regional Retail Lead"],
    "logistics": ["Dispatcher", "Fleet Coordinator", "Warehouse Supervisor", "Logistics Analyst"],
    "agricore": ["Field Agent", "Processing Facility Supervisor", "Quality Inspector", "Agronomist"],
    "properties": ["Facilities Manager", "Leasing Officer", "Maintenance Coordinator"],
    "vfs": ["Loan Officer", "Compliance Analyst", "Wallet Operations Lead"],
    "group": ["Group Executive", "Group Finance", "Group HR"],
}

def generate_employees():
    print("Generating employees...")
    rows = []
    division_of_employee = []
    # Assign divisions up front so reports_to can reference an earlier id
    # in the SAME division (a plausible manager, not a random stranger).
    counts = {div: int(round(N_EMPLOYEES * w)) for div, w in DIVISION_EMPLOYEE_WEIGHTS.items()}
    # AgriCore needs at least ~140 * SCALE field agents for realism (brief 1.3.4)
    counts["agricore"] = max(counts["agricore"], scaled(140, floor=5))
    # fix rounding drift against N_EMPLOYEES
    diff = N_EMPLOYEES - sum(counts.values())
    counts["retail"] += diff

    eid = 1
    first_id_of_division = {}
    for div, cnt in counts.items():
        first_id_of_division[div] = eid
        for i in range(cnt):
            name = fake.name()
            role = random.choice(ROLE_TITLES_BY_DIVISION[div])
            status = random.choices(["active", "on_leave", "terminated"], weights=[0.90, 0.05, 0.05])[0]
            hire = random_date(date(2016, 1, 1), DATE_END)
            # 15% chance of reporting to an earlier employee in the same division
            reports_to = None
            if i > 3 and random.random() < 0.85:
                reports_to = random.randint(first_id_of_division[div], eid - 1)
            rows.append((eid, name, div, role, status, hire.isoformat(),
                         reports_to if reports_to else "", datetime.now().isoformat()))
            division_of_employee.append(div)
            eid += 1
    write_csv("employees", [
        "employee_id", "full_name", "division_id", "role_title",
        "employment_status", "hire_date", "reports_to", "created_date"
    ], rows)
    return division_of_employee  # index 0 == employee_id 1

SUPPLIER_TYPE_WEIGHTS_NONFARMER = {"goods_vendor": 0.45, "service_vendor": 0.30, "logistics_partner": 0.25}

def generate_suppliers():
    print("Generating suppliers_vendors...")
    rows = []
    supplier_types = []
    n_farmers = int(round(N_SUPPLIERS * PCT_SUPPLIERS_ARE_FARMERS))
    for sid in range(1, N_SUPPLIERS + 1):
        if sid <= n_farmers:
            stype = "farmer"
            primary_division = "agricore"
        else:
            stype = random.choices(list(SUPPLIER_TYPE_WEIGHTS_NONFARMER.keys()),
                                    weights=list(SUPPLIER_TYPE_WEIGHTS_NONFARMER.values()))[0]
            primary_division = {"goods_vendor": "retail", "service_vendor": "properties",
                                 "logistics_partner": "logistics"}[stype]
        supplier_types.append(stype)
        legal_name = fake.company() if stype != "farmer" else f"{fake.last_name()} Farms"
        onboarding = random_date(date(2017, 1, 1), DATE_END)
        status = random.choices(["active", "inactive", "suspended"], weights=[0.90, 0.07, 0.03])[0]
        rows.append((sid, legal_name, stype, primary_division, onboarding.isoformat(), status))
    # shuffle so "farmer" rows aren't all clustered at the low ids only --
    # actually keep them low-numbered on purpose, it makes farms.csv join
    # logic below simpler to read; realism doesn't depend on id ordering.
    write_csv("suppliers_vendors", [
        "supplier_id", "legal_name", "supplier_type", "primary_division_id",
        "onboarding_date", "status"
    ], rows)
    return supplier_types  # index 0 == supplier_id 1

def generate_locations(n_needed_by_type):
    """n_needed_by_type: dict like {'store': 68, 'warehouse': 6, ...}"""
    print("Generating locations_sites...")
    rows = []
    lid = 1
    location_ids_by_type = {t: [] for t in n_needed_by_type}
    for site_type, count in n_needed_by_type.items():
        for _ in range(count):
            country = pick_country()
            city = random.choice(CITY_BY_COUNTRY[country])
            name_map = {
                "store": f"Meridian Mart {city} {lid}",
                "warehouse": f"Concord Depot {city} {lid}",
                "farm": f"{city} Farm Site {lid}",
                "property": f"Veridian Property {city} {lid}",
                "office": f"Veridian Office {city} {lid}",
                "other": f"Veridian Site {city} {lid}",
            }
            lat = round(random.uniform(4.0, 13.5), 6)   # rough West/East Africa band
            lon = round(random.uniform(-3.5, 41.0), 6)
            rows.append((lid, name_map[site_type], site_type, fake.street_address(),
                         city, COUNTRY_CODE[country], lat, lon))
            location_ids_by_type[site_type].append(lid)
            lid += 1
    write_csv("locations_sites", [
        "location_id", "site_name", "site_type", "address", "city",
        "country_code", "latitude", "longitude"
    ], rows)
    return location_ids_by_type

def generate_products():
    print("Generating product_service_catalogue...")
    retail_categories = ["Grains & Staples", "Fresh Produce", "Household Goods", "Beverages", "Personal Care"]
    agri_categories = ["Plantain", "Maize", "Cassava", "Yam", "Rice (Paddy)", "Cocoa", "Groundnut"]
    rows = []
    n_agri = int(round(N_PRODUCTS * 0.35))
    agri_product_ids = list(range(1, n_agri + 1))
    for pid in range(1, N_PRODUCTS + 1):
        if pid <= n_agri:
            cat = random.choice(agri_categories)
            name = f"{cat} - {fake.word().capitalize()} grade"
            uom = "kg"
            division = "agricore"
        else:
            cat = random.choice(retail_categories)
            name = f"{fake.word().capitalize()} {cat.split()[0]}"
            uom = random.choice(["unit", "kg", "litre", "pack"])
            division = "retail"
        rows.append((pid, name, cat, uom, division, random.random() < 0.95))
    write_csv("product_service_catalogue", [
        "product_id", "product_name", "category", "unit_of_measure",
        "primary_division_id", "is_active"
    ], rows)
    return agri_product_ids

def generate_financial_account_references(planned_wallet_owners):
    """planned_wallet_owners: list of ('customer'|'supplier', id) tuples,
    one per wallet we intend to create. Returns list of account_ref_id
    aligned to that same order."""
    print("Generating financial_account_references...")
    rows = []
    ref_ids = []
    for i, (owner_type, owner_id) in enumerate(planned_wallet_owners, start=1):
        customer_id = owner_id if owner_type == "customer" else ""
        supplier_id = owner_id if owner_type == "supplier" else ""
        opened = random_date(date(2019, 1, 1), DATE_END)
        rows.append((i, customer_id, supplier_id, "wallet", "active", opened.isoformat()))
        ref_ids.append(i)
    write_csv("financial_account_references", [
        "account_ref_id", "customer_id", "supplier_id", "account_type",
        "account_status", "opened_date"
    ], rows)
    return ref_ids

# ============================================================================
# 2. VERIDIAN PROPERTIES
# ============================================================================

def generate_properties(property_location_ids):
    print("Generating properties...")
    rows = []
    property_types = []
    for i, loc_id in enumerate(property_location_ids, start=1):
        ptype = random.choices(
            ["retail_unit", "warehouse", "office", "farmland", "mixed_use", "other"],
            weights=[0.35, 0.15, 0.10, 0.10, 0.20, 0.10])[0]
        size = round(random.uniform(80, 5000), 2)
        ownership = random.choices(["owned", "leased_in"], weights=[0.6, 0.4])[0]
        rows.append((i, loc_id, ptype, size, ownership))
        property_types.append(ptype)
    write_csv("properties", ["property_id", "location_id", "property_type", "size_sqm", "ownership_status"], rows)
    return property_types

def generate_tenants(n_internal_tenants=6):
    """One internal tenant per division that leases space (retail, logistics,
    agricore all lease from Properties per brief 1.4/1.3.5), plus a pool of
    external commercial/residential tenants."""
    print("Generating tenants...")
    rows = []
    tid = 1
    internal_divs = ["retail", "logistics", "agricore"]
    tenant_kind = []
    for div in internal_divs:
        rows.append((tid, "internal", div, ""))
        tenant_kind.append(("internal", div))
        tid += 1
    n_external = scaled(60, floor=5)
    for _ in range(n_external):
        name = fake.company()
        rows.append((tid, "external", "", name))
        tenant_kind.append(("external", None))
        tid += 1
    write_csv("tenants", ["tenant_id", "tenant_type", "division_id", "external_tenant_name"], rows)
    return tenant_kind  # index 0 == tenant_id 1

def generate_leases(n_properties, n_tenants):
    print("Generating leases...")
    rows = []
    # Ensure "at most one ACTIVE lease per tenant" holds: track which
    # tenants already have an active lease as we assign them.
    tenant_has_active = set()
    lid = 1
    tenants_pool = list(range(1, n_tenants + 1))
    random.shuffle(tenants_pool)
    for _ in range(N_LEASES):
        prop_id = random.randint(1, n_properties)
        # try a few times to find a tenant without an active lease yet,
        # otherwise just create an expired/terminated one for that tenant
        tenant_id = random.choice(tenants_pool)
        start = random_date(date(2018, 1, 1), DATE_END - timedelta(days=30))
        end = start + timedelta(days=random.choice([180, 365, 365 * 2, 365 * 3]))
        if tenant_id not in tenant_has_active and end > date.today():
            status = "active"
            tenant_has_active.add(tenant_id)
        elif end < date.today():
            status = random.choices(["expired", "terminated"], weights=[0.85, 0.15])[0]
        else:
            status = "terminated"  # would overlap an existing active lease otherwise
        rent = round(random.uniform(50_000, 3_000_000), 2)  # NGN-scale rents
        rows.append((lid, prop_id, tenant_id, start.isoformat(), end.isoformat(), rent, status))
        lid += 1
    write_csv("leases", ["lease_id", "property_id", "tenant_id", "start_date", "end_date", "monthly_rent", "status"], rows)

def generate_property_maintenance_and_assets(n_properties):
    print("Generating maintenance_requests, property_valuations, utility_accounts, facility_assets...")
    # maintenance_requests
    rows = []
    for i in range(1, N_MAINTENANCE_REQUESTS + 1):
        prop_id = random.randint(1, n_properties)
        requested = random_date(date(2023, 1, 1), DATE_END)
        category = random.choice(["electrical", "plumbing", "structural", "security", "cleaning", "other"])
        status = random.choices(["open", "in_progress", "resolved", "cancelled"], weights=[0.15, 0.15, 0.6, 0.1])[0]
        resolved = ""
        if status == "resolved":
            resolved = (requested + timedelta(days=random.randint(1, 45))).isoformat()
        rows.append((i, prop_id, requested.isoformat(), category, status, resolved))
    write_csv("maintenance_requests", ["request_id", "property_id", "requested_date", "category", "status", "resolved_date"], rows)

    # property_valuations -- roughly one per property per year since 2022
    rows = []
    vid = 1
    for prop_id in range(1, n_properties + 1):
        base_value = random.uniform(15_000_000, 800_000_000)
        for yr in (2022, 2023, 2024, 2025):
            val = round(base_value * random.uniform(0.95, 1.15) ** (yr - 2022), 2)
            rows.append((vid, prop_id, date(yr, random.randint(1, 12), 1).isoformat(), val))
            vid += 1
    write_csv("property_valuations", ["valuation_id", "property_id", "valuation_date", "assessed_value"], rows)

    # utility_accounts -- 1-3 utilities per property
    rows = []
    uid = 1
    providers = {"electricity": ["Ikeja Electric", "AEDC", "ECG"], "water": ["State Water Board"],
                 "gas": ["Nigerian Gas Co"], "internet": ["MTN Business", "Airtel Business"], "waste": ["Cleanserve Ltd"]}
    for prop_id in range(1, n_properties + 1):
        n_util = random.randint(1, 3)
        for utype in random.sample(list(providers.keys()), n_util):
            rows.append((uid, prop_id, utype, random.choice(providers[utype]), fake.bothify("ACC-########")))
            uid += 1
    write_csv("utility_accounts", ["utility_id", "property_id", "utility_type", "provider_name", "account_number"], rows)

    # facility_assets -- 2-6 assets per property
    rows = []
    aid = 1
    asset_types = ["Generator", "HVAC Unit", "CCTV System", "Fire Suppression System", "Elevator", "Solar Array"]
    for prop_id in range(1, n_properties + 1):
        for _ in range(random.randint(2, 6)):
            installed = random_date(date(2015, 1, 1), DATE_END)
            rows.append((aid, prop_id, random.choice(asset_types), installed.isoformat(),
                        random.choices(["excellent", "good", "fair", "poor"], weights=[0.2, 0.45, 0.25, 0.1])[0]))
            aid += 1
    write_csv("facility_assets", ["asset_id", "property_id", "asset_type", "installed_date", "condition_rating"], rows)

# ============================================================================
# 3. CONCORD LOGISTICS
# ============================================================================

def generate_vehicles():
    print("Generating vehicles...")
    rows = []
    for vid in range(1, N_VEHICLES + 1):
        vtype = random.choices(["van", "truck", "trailer", "motorbike"], weights=[0.35, 0.35, 0.15, 0.15])[0]
        capacity = {"van": (500, 2000), "truck": (3000, 15000), "trailer": (15000, 30000), "motorbike": (20, 80)}[vtype]
        rows.append((vid, fake.bothify("??-####-??").upper(), vtype,
                    round(random.uniform(*capacity), 2),
                    random.choices(["active", "in_maintenance", "retired"], weights=[0.85, 0.10, 0.05])[0]))
    write_csv("vehicles", ["vehicle_id", "registration_number", "vehicle_type", "capacity_kg", "status"], rows)

def generate_drivers(division_of_employee):
    print("Generating drivers...")
    logistics_employee_ids = [i + 1 for i, d in enumerate(division_of_employee) if d == "logistics"]
    n_drivers = min(len(logistics_employee_ids), scaled(250, floor=5))
    rows = []
    chosen = random.sample(logistics_employee_ids, n_drivers) if logistics_employee_ids else []
    for did, emp_id in enumerate(chosen, start=1):
        expiry = random_date(date(2026, 1, 1), date(2029, 12, 31))
        rows.append((did, emp_id, fake.bothify("DL-#######"), expiry.isoformat()))
    # a handful of contracted (non-employee) drivers too
    n_contracted = scaled(20, floor=2)
    for j in range(n_contracted):
        did = n_drivers + j + 1
        expiry = random_date(date(2026, 1, 1), date(2029, 12, 31))
        rows.append((did, "", fake.bothify("DL-#######"), expiry.isoformat()))
    write_csv("drivers", ["driver_id", "employee_id", "licence_number", "licence_expiry"], rows)
    return n_drivers + n_contracted

def generate_shipments(all_location_ids):
    print("Generating shipments...")
    rows = []
    for sid in range(1, N_SHIPMENTS + 1):
        origin, dest = random.sample(all_location_ids, 2)
        client_type = random.choices(["internal", "external"], weights=[0.55, 0.45])[0]  # brief 1.3.2
        status = random.choices(["scheduled", "in_transit", "delivered", "delayed", "cancelled"],
                                weights=[0.10, 0.15, 0.65, 0.07, 0.03])[0]
        rows.append((sid, origin, dest, client_type, status))
    write_csv("shipments", ["shipment_id", "origin_location_id", "destination_location_id", "client_type", "status"], rows)

def generate_shipment_legs(n_shipments, n_vehicles, n_drivers):
    print("Generating shipment_legs...")
    rows = []
    leg_id = 1
    for shipment_id in range(1, n_shipments + 1):
        n_legs = random.choices([1, 2], weights=[0.8, 0.2])[0]  # most shipments are direct
        for _ in range(n_legs):
            departure = datetime.combine(random_date(), datetime.min.time()) + timedelta(hours=random.randint(5, 20))
            arrival = departure + timedelta(hours=random.uniform(1, 30))
            rows.append((leg_id, shipment_id, random.randint(1, n_vehicles), random.randint(1, n_drivers),
                        departure.isoformat(), arrival.isoformat()))
            leg_id += 1
    write_csv("shipment_legs", ["leg_id", "shipment_id", "vehicle_id", "driver_id", "departure_time", "arrival_time"], rows)

def generate_routes(all_location_ids):
    print("Generating routes...")
    rows = []
    for rid in range(1, N_ROUTES + 1):
        origin, dest = random.sample(all_location_ids, 2)
        rows.append((rid, f"Route {rid:03d}", origin, dest, round(random.uniform(15, 950), 2)))
    write_csv("routes", ["route_id", "route_name", "origin_location_id", "destination_location_id", "distance_km"], rows)

def generate_warehouses(warehouse_location_ids, property_ids_that_are_warehouses):
    print("Generating warehouses...")
    rows = []
    for i, loc_id in enumerate(warehouse_location_ids, start=1):
        leased_from = random.choice(property_ids_that_are_warehouses) if (property_ids_that_are_warehouses and random.random() < 0.7) else ""
        rows.append((i, loc_id, random.randint(2000, 50000), leased_from))
    write_csv("warehouses", ["warehouse_id", "location_id", "capacity_units", "leased_from_property_id"], rows)

def generate_maintenance_logs(n_vehicles):
    print("Generating maintenance_logs...")
    rows = []
    mid = 1
    for vehicle_id in range(1, n_vehicles + 1):
        for _ in range(random.randint(1, 5)):
            svc_date = random_date(date(2023, 1, 1), DATE_END)
            rows.append((mid, vehicle_id, svc_date.isoformat(), round(random.uniform(15_000, 900_000), 2),
                        random.choice(["Routine service", "Brake replacement", "Tyre replacement",
                                      "Engine repair", "Electrical fault", "Bodywork"])))
            mid += 1
    write_csv("maintenance_logs", ["maintenance_id", "vehicle_id", "service_date", "cost", "description"], rows)

# ============================================================================
# 4. VERIDIAN FINANCIAL SERVICES
#    Narrowest-write-access module (brief 4.4) -- generated as a curated,
#    externally-sourced view, same as it would be loaded in production.
# ============================================================================

def generate_wallet_accounts(wallet_owners, account_ref_ids):
    """wallet_owners: list of ('customer'|'supplier', id), same order/length
    as account_ref_ids (one financial_account_references row per wallet)."""
    print("Generating wallet_accounts...")
    rows = []
    for i, ((owner_type, owner_id), ref_id) in enumerate(zip(wallet_owners, account_ref_ids), start=1):
        customer_id = owner_id if owner_type == "customer" else ""
        supplier_id = owner_id if owner_type == "supplier" else ""
        balance = round(random.uniform(0, 500_000), 2)
        status = random.choices(["active", "frozen", "closed"], weights=[0.92, 0.03, 0.05])[0]
        rows.append((i, customer_id, supplier_id, ref_id, balance, status))
    write_csv("wallet_accounts", ["wallet_id", "customer_id", "supplier_id", "account_ref_id", "balance", "status"], rows)
    return len(rows)

def generate_wallet_transactions(n_wallets):
    print("Generating wallet_transactions...")
    # brief 5.4: "approximately one to two months of realistic wallet activity"
    window_start = DATE_END - timedelta(days=60)
    wallet_ids = np.random.randint(1, n_wallets + 1, size=N_WALLET_TRANSACTIONS)
    counterparty_types = np.random.choice(
        ["merchant", "peer", "vfs_internal", "external_bank"],
        size=N_WALLET_TRANSACTIONS, p=[0.55, 0.25, 0.10, 0.10])
    amounts = np.round(np.random.exponential(scale=8000, size=N_WALLET_TRANSACTIONS) + 200, 2)
    dates = random_dates_vectorized(N_WALLET_TRANSACTIONS, window_start, DATE_END)
    rows = []
    for i in range(N_WALLET_TRANSACTIONS):
        t = datetime.combine(dates[i], datetime.min.time()) + timedelta(
            hours=int(np.random.randint(6, 23)), minutes=int(np.random.randint(0, 59)))
        rows.append((i + 1, int(wallet_ids[i]), counterparty_types[i], float(amounts[i]), t.isoformat()))
    write_csv("wallet_transactions", ["wallet_txn_id", "wallet_id", "counterparty_type", "amount", "transaction_date"], rows)

def generate_loans(farmer_supplier_ids, merchant_customer_ids):
    """Two borrower populations, matching brief 1.3.3: smallholder farmers
    and small retail merchants. Returns (loans_list, supplier_to_loan_ids)
    for use by farmer_loans_reference later."""
    print("Generating loans...")
    rows = []
    loans_list = []  # (loan_id, principal, status) for the repayments generator
    supplier_to_loan_ids = {}
    loan_id = 1
    for sid in farmer_supplier_ids:
        principal = round(random.uniform(50_000, 3_000_000), 2)
        status = random.choices(["active", "repaid", "defaulted", "written_off"], weights=[0.55, 0.30, 0.10, 0.05])[0]
        rows.append((loan_id, "", sid, principal, status))
        loans_list.append((loan_id, principal, status))
        supplier_to_loan_ids.setdefault(sid, []).append(loan_id)
        loan_id += 1
    for cid in merchant_customer_ids:
        principal = round(random.uniform(100_000, 5_000_000), 2)
        status = random.choices(["active", "repaid", "defaulted", "written_off"], weights=[0.55, 0.30, 0.10, 0.05])[0]
        rows.append((loan_id, cid, "", principal, status))
        loans_list.append((loan_id, principal, status))
        loan_id += 1
    write_csv("loans", ["loan_id", "borrower_customer_id", "borrower_supplier_id", "principal_amount", "status"], rows)
    return loans_list, supplier_to_loan_ids

def generate_loan_repayments(loans_list):
    print("Generating loan_repayments...")
    rows = []
    rid = 1
    today = date.today()
    for loan_id, principal, status in loans_list:
        n_installments = random.randint(6, 24)
        installment_amt = round(principal / n_installments, 2)
        start = random_date(date(2023, 1, 1), date(2025, 6, 1))
        for i in range(n_installments):
            due = start + timedelta(days=30 * i)
            if status == "repaid" or (due < today and random.random() < 0.85):
                paid = installment_amt
                paid_date = (due + timedelta(days=random.randint(-3, 12))).isoformat()
            elif status == "defaulted" and due > start + timedelta(days=90):
                paid, paid_date = 0, ""
            else:
                paid, paid_date = 0, ""
            rows.append((rid, loan_id, due.isoformat(), installment_amt, paid, paid_date))
            rid += 1
    write_csv("loan_repayments", ["repayment_id", "loan_id", "due_date", "amount_due", "amount_paid", "paid_date"], rows)

def generate_kyc_records(customer_ids_needing_kyc):
    print("Generating kyc_records...")
    rows = []
    for i, cid in enumerate(sorted(set(customer_ids_needing_kyc)), start=1):
        level = random.choices(["tier1", "tier2", "tier3"], weights=[0.5, 0.35, 0.15])[0]
        verified = random_date(date(2019, 1, 1), DATE_END) if random.random() < 0.9 else None
        rows.append((i, cid, level, verified.isoformat() if verified else ""))
    write_csv("kyc_records", ["kyc_id", "customer_id", "verification_level", "verified_date"], rows)

def generate_merchant_settlements():
    print("Generating merchant_settlements...")
    rows = []
    sid = 1
    for div in ["retail", "logistics", "agricore", "properties"]:
        n = scaled(200, floor=10)
        for _ in range(n):
            dt = random_date(date(2025, 1, 1), DATE_END)
            rows.append((sid, div, dt.isoformat(), round(random.uniform(200_000, 50_000_000), 2)))
            sid += 1
    write_csv("merchant_settlements", ["settlement_id", "division_id", "settlement_date", "total_amount"], rows)

# ============================================================================
# 5. MERIDIAN RETAIL AND CONSUMER
# ============================================================================

def generate_stores(store_location_ids, employee_ids_by_division):
    print("Generating stores...")
    rows = []
    retail_employees = employee_ids_by_division.get("retail", [])
    for i, loc_id in enumerate(store_location_ids, start=1):
        fmt = random.choices(["flagship", "standard", "express", "online"], weights=[0.08, 0.55, 0.30, 0.07])[0]
        opening = random_date(date(2010, 1, 1), date(2025, 1, 1))
        manager = random.choice(retail_employees) if retail_employees else ""
        rows.append((i, loc_id, fmt, opening.isoformat(), manager))
    write_csv("stores", ["store_id", "location_id", "store_format", "opening_date", "manager_employee_id"], rows)

# Retail sees roughly 38,000 POS transactions/day across 68 stores (brief
# 1.3.1) -- i.e. traffic is heavier on weekends. This weight vector is
# applied to plausible transaction timestamps rather than spreading them
# uniformly across the week.
WEEKDAY_TRAFFIC_WEIGHTS = [0.12, 0.12, 0.12, 0.13, 0.15, 0.20, 0.16]  # Mon..Sun

def generate_pos_transactions(n_stores, customer_ids_pool):
    print("Generating pos_transactions...")
    window_start = DATE_END - timedelta(days=60)  # brief 5.4: "one to two months"
    n_days = (DATE_END - window_start).days + 1
    day_offsets = np.arange(n_days)
    day_dates = [window_start + timedelta(days=int(d)) for d in day_offsets]
    day_weights = np.array([WEEKDAY_TRAFFIC_WEIGHTS[d.weekday()] for d in day_dates])
    day_weights = day_weights / day_weights.sum()
    chosen_days = np.random.choice(len(day_dates), size=N_POS_TRANSACTIONS, p=day_weights)

    store_ids = np.random.randint(1, n_stores + 1, size=N_POS_TRANSACTIONS)
    # 55% of transactions have a known loyalty/registered customer, 45% walk-in
    has_customer = np.random.random(N_POS_TRANSACTIONS) < 0.55
    payment_methods = np.random.choice(["cash", "card", "wallet", "other"], size=N_POS_TRANSACTIONS, p=[0.30, 0.25, 0.40, 0.05])
    amounts = np.round(np.random.exponential(scale=6500, size=N_POS_TRANSACTIONS) + 500, 2)

    rows = []
    for i in range(N_POS_TRANSACTIONS):
        d = day_dates[chosen_days[i]]
        t = datetime.combine(d, datetime.min.time()) + timedelta(hours=int(np.random.randint(7, 22)), minutes=int(np.random.randint(0, 59)))
        cust = random.choice(customer_ids_pool) if has_customer[i] and customer_ids_pool else ""
        rows.append((i + 1, int(store_ids[i]), cust, t.isoformat(), float(amounts[i]), payment_methods[i]))
    write_csv("pos_transactions", ["transaction_id", "store_id", "customer_id", "transaction_date", "total_amount", "payment_method"], rows)

def generate_transaction_line_items(n_transactions, n_products):
    print("Generating transaction_line_items...")
    rows = []
    lid = 1
    # 1-5 line items per transaction, vectorized in chunks for speed
    n_items_per_txn = np.random.randint(1, 6, size=n_transactions)
    product_ids_flat = np.random.randint(1, n_products + 1, size=int(n_items_per_txn.sum()))
    quantities_flat = np.random.randint(1, 6, size=int(n_items_per_txn.sum()))
    unit_prices_flat = np.round(np.random.uniform(100, 15000, size=int(n_items_per_txn.sum())), 2)
    ptr = 0
    for txn_id in range(1, n_transactions + 1):
        k = n_items_per_txn[txn_id - 1]
        for j in range(k):
            rows.append((lid, txn_id, int(product_ids_flat[ptr]), int(quantities_flat[ptr]), float(unit_prices_flat[ptr])))
            lid += 1
            ptr += 1
    write_csv("transaction_line_items", ["line_item_id", "transaction_id", "product_id", "quantity", "unit_price"], rows)

def generate_inventory_stock_levels(n_stores, n_products):
    print("Generating inventory_stock_levels...")
    rows = []
    sid = 1
    for store_id in range(1, n_stores + 1):
        # each store stocks a random subset of the catalogue, not all of it
        n_products_stocked = random.randint(int(n_products * 0.4), n_products)
        for product_id in random.sample(range(1, n_products + 1), n_products_stocked):
            qty = random.randint(0, 500)
            counted = random_date(DATE_END - timedelta(days=14), DATE_END)
            rows.append((sid, store_id, product_id, qty, counted.isoformat()))
            sid += 1
    write_csv("inventory_stock_levels", ["stock_id", "store_id", "product_id", "quantity_on_hand", "last_counted_date"], rows)

def generate_promotions(n_products):
    print("Generating promotions...")
    rows = []
    n_promos = scaled(150, floor=10)
    for i in range(1, n_promos + 1):
        product_id = random.randint(1, n_products)
        dtype = random.choice(["percentage", "fixed_amount", "bogo"])
        value = round(random.uniform(5, 30), 2) if dtype == "percentage" else round(random.uniform(50, 2000), 2)
        start = random_date(date(2025, 1, 1), DATE_END)
        end = start + timedelta(days=random.randint(3, 30))
        rows.append((i, product_id, dtype, value, start.isoformat(), end.isoformat()))
    write_csv("promotions", ["promotion_id", "product_id", "discount_type", "discount_value", "start_date", "end_date"], rows)

def generate_loyalty_accounts(loyalty_customer_ids):
    print("Generating loyalty_accounts...")
    rows = []
    lid = 1
    for cid in loyalty_customer_ids:
        # 5.3 cardinality: at most one loyalty account per store format --
        # most customers just have one (their preferred/nearest format);
        # a smaller share are enrolled under two formats (e.g. standard + online).
        n_formats = random.choices([1, 2], weights=[0.85, 0.15])[0]
        formats = random.sample(["flagship", "standard", "express", "online"], n_formats)
        for fmt in formats:
            tier = random.choices(["bronze", "silver", "gold", "platinum"], weights=[0.55, 0.28, 0.13, 0.04])[0]
            points = random.randint(0, 50_000)
            enrolled = random_date(date(2019, 1, 1), DATE_END)
            rows.append((lid, cid, fmt, tier, points, enrolled.isoformat()))
            lid += 1
    write_csv("loyalty_accounts", ["loyalty_id", "customer_id", "store_format", "tier", "points_balance", "enrolment_date"], rows)

def generate_supplier_deliveries(non_farmer_retail_supplier_ids, n_stores):
    print("Generating supplier_deliveries...")
    rows = []
    n_deliveries = scaled(4000, floor=30)
    pool = non_farmer_retail_supplier_ids if non_farmer_retail_supplier_ids else [1]
    for i in range(1, n_deliveries + 1):
        supplier_id = random.choice(pool)
        store_id = random.randint(1, n_stores)
        expected = random_date(DATE_END - timedelta(days=90), DATE_END)
        status = random.choices(["pending", "in_transit", "received", "cancelled"], weights=[0.10, 0.10, 0.75, 0.05])[0]
        received = ""
        if status == "received":
            received = (expected + timedelta(days=random.randint(-1, 5))).isoformat()
        rows.append((i, supplier_id, store_id, expected.isoformat(), received, status))
    write_csv("supplier_deliveries", ["delivery_id", "supplier_id", "store_id", "expected_date", "received_date", "status"], rows)

# ============================================================================
# 6. AGRICORE
#    Full traceability chain: Farm -> Harvest Batch -> Processing Run ->
#    Quality Grade -> Wholesale Shipment (brief 5.3). Harvest volume gets
#    a seasonal curve rather than uniform-random dates (Appendix D).
# ============================================================================

# Simple bimodal seasonal curve (two harvest windows across the year,
# common for staples like maize/cassava in the brief's West African
# setting) -- documented assumption, not a claim about real agronomy.
MONTH_HARVEST_WEIGHTS = [0.05, 0.05, 0.08, 0.12, 0.14, 0.10, 0.05, 0.05, 0.09, 0.13, 0.10, 0.04]

def plan_farms_per_farmer(farmer_supplier_ids):
    """80% of farmers register 1 farm, 15% register 2, 5% register 3
    (brief 5.3: 'common among AgriCore's larger cooperative-affiliated
    growers' to hold multiple farms)."""
    plan = {}
    for sid in farmer_supplier_ids:
        n = random.choices([1, 2, 3], weights=[0.80, 0.15, 0.05])[0]
        plan[sid] = n
    return plan

def generate_farms(farms_plan, farm_location_ids):
    print("Generating farms...")
    crops = ["Plantain", "Maize", "Cassava", "Yam", "Rice (Paddy)", "Cocoa", "Groundnut"]
    rows = []
    fid = 1
    loc_iter = iter(farm_location_ids)
    supplier_to_farm_ids = {}
    for supplier_id, n_farms in farms_plan.items():
        supplier_to_farm_ids[supplier_id] = []
        for _ in range(n_farms):
            loc_id = next(loc_iter)
            size = round(random.uniform(0.5, 25.0), 2)
            rows.append((fid, supplier_id, loc_id, size, random.choice(crops)))
            supplier_to_farm_ids[supplier_id].append(fid)
            fid += 1
    write_csv("farms", ["farm_id", "supplier_id", "location_id", "size_hectares", "primary_crop"], rows)
    return supplier_to_farm_ids

def generate_farmers(farmer_supplier_ids):
    print("Generating farmers...")
    rows = []
    supplier_to_farmer_id = {}
    coop_names = [f"{c} Farmers Cooperative" for c in ["Uyo", "Kaduna", "Ibadan", "Kumasi", "Nakuru", "Enugu", "Benue"]]
    for i, sid in enumerate(farmer_supplier_ids, start=1):
        reg = random_date(date(2015, 1, 1), DATE_END)
        coop = random.choice(coop_names) if random.random() < 0.6 else ""
        rows.append((i, sid, reg.isoformat(), coop))
        supplier_to_farmer_id[sid] = i
    write_csv("farmers", ["farmer_id", "supplier_id", "registration_date", "cooperative_name"], rows)
    return supplier_to_farmer_id

def generate_harvest_batches(supplier_to_farm_ids, agri_product_ids, agricore_field_agent_ids):
    print("Generating harvest_batches...")
    all_farm_ids = [fid for farm_list in supplier_to_farm_ids.values() for fid in farm_list]
    rows = []
    months = list(range(1, 13))
    for hid in range(1, N_HARVEST_BATCHES + 1):
        farm_id = random.choice(all_farm_ids)
        product_id = random.choice(agri_product_ids)
        yr = random.choice([2024, 2025, 2026])
        month = random.choices(months, weights=MONTH_HARVEST_WEIGHTS)[0]
        day = random.randint(1, 28)
        h_date = date(yr, month, day)
        if h_date > DATE_END:
            h_date = DATE_END
        volume = round(random.uniform(200, 8000), 2)
        agent = random.choice(agricore_field_agent_ids) if agricore_field_agent_ids else ""
        rows.append((hid, farm_id, product_id, h_date.isoformat(), volume, agent))
    write_csv("harvest_batches", ["harvest_id", "farm_id", "product_id", "harvest_date", "volume_kg", "field_agent_employee_id"], rows)

def generate_processing_runs(n_harvest_batches, facility_location_ids):
    print("Generating processing_runs...")
    rows = []
    run_id = 1
    harvest_to_runs = {}
    for harvest_id in range(1, n_harvest_batches + 1):
        # brief 5.3: zero, one, or more processing runs per harvest batch
        n_runs = random.choices([0, 1, 2], weights=[0.10, 0.75, 0.15])[0]
        harvest_to_runs[harvest_id] = []
        for _ in range(n_runs):
            facility = random.choice(facility_location_ids)
            run_date = random_date(date(2024, 1, 1), DATE_END)
            output = round(random.uniform(150, 7500), 2)
            rows.append((run_id, harvest_id, facility, run_date.isoformat(), output))
            harvest_to_runs[harvest_id].append(run_id)
            run_id += 1
    write_csv("processing_runs", ["run_id", "harvest_id", "facility_location_id", "run_date", "output_volume_kg"], rows)
    return run_id - 1

def generate_quality_grades(n_runs, agricore_inspector_ids):
    print("Generating quality_grades...")
    rows = []
    for i in range(1, n_runs + 1):
        grade = random.choices(["A", "B", "C", "reject"], weights=[0.40, 0.35, 0.20, 0.05])[0]
        moisture = round(random.uniform(8, 18), 2)
        inspector = random.choice(agricore_inspector_ids) if agricore_inspector_ids else ""
        rows.append((i, i, grade, moisture, inspector))
    write_csv("quality_grades", ["grade_id", "run_id", "grade_level", "moisture_content", "inspector_employee_id"], rows)

def generate_wholesale_shipments(n_runs, n_stores, n_shipments):
    print("Generating wholesale_shipments...")
    rows = []
    wid = 1
    for run_id in range(1, n_runs + 1):
        if random.random() < 0.7:  # not every run has shipped out yet
            dest_type = random.choices(["meridian_retail", "external_client"], weights=[0.65, 0.35])[0]
            dest_id = random.randint(1, n_stores) if dest_type == "meridian_retail" else random.randint(1, 50)
            linked_shipment = random.randint(1, n_shipments) if random.random() < 0.6 else ""
            rows.append((wid, run_id, dest_type, dest_id, linked_shipment))
            wid += 1
    write_csv("wholesale_shipments", ["wholesale_id", "run_id", "destination_type", "destination_id", "shipment_id"], rows)

def generate_farmer_loans_reference(supplier_to_farmer_id, supplier_to_loan_ids, loan_status_by_id):
    print("Generating farmer_loans_reference...")
    status_map = {"active": "current", "repaid": "closed", "defaulted": "overdue", "written_off": "closed"}
    rows = []
    rid = 1
    for supplier_id, loan_ids in supplier_to_loan_ids.items():
        farmer_id = supplier_to_farmer_id.get(supplier_id)
        if farmer_id is None:
            continue
        for loan_id in loan_ids:
            visible_status = status_map[loan_status_by_id[loan_id]]
            rows.append((rid, farmer_id, loan_id, visible_status))
            rid += 1
    write_csv("farmer_loans_reference", ["reference_id", "farmer_id", "loan_id", "visible_summary_status"], rows)

# ============================================================================
# ORCHESTRATION
# ============================================================================

PCT_CUSTOMERS_WITH_LOYALTY = 0.44          # brief 1.3.1: ~610k / ~1.4M real ratio
PCT_NONLOYALTY_CUSTOMERS_WITH_WALLET = 0.15
PCT_FARMERS_WITH_WALLET = 0.20

def main():
    print(f"=== Project Concord synthetic data generator (SCALE={SCALE}) ===")
    print(f"Output directory: {os.path.abspath(OUT_DIR)}\n")

    # ---- 1. Core Hub identity pools ----
    countries = generate_customers()
    all_customer_ids = list(range(1, N_CUSTOMERS + 1))

    division_of_employee = generate_employees()
    employee_ids_by_division = {d: [] for d in DIVISION_CODES}
    for idx, div in enumerate(division_of_employee, start=1):
        employee_ids_by_division[div].append(idx)

    supplier_types = generate_suppliers()
    farmer_supplier_ids = [i + 1 for i, t in enumerate(supplier_types) if t == "farmer"]
    non_farmer_delivery_supplier_ids = [i + 1 for i, t in enumerate(supplier_types) if t in ("farmer", "goods_vendor")]

    # Plan farm counts now (needed to size the location pool) before any
    # locations are actually generated.
    farms_plan = plan_farms_per_farmer(farmer_supplier_ids)
    total_farms = sum(farms_plan.values())

    n_needed_by_type = {
        "store": N_STORES,
        "warehouse": N_WAREHOUSES,
        "farm": total_farms,
        "property": N_PROPERTIES,
        "office": scaled(8, floor=2),
        "other": N_PROCESSING_FACILITIES + scaled(5, floor=1),  # processing facilities + misc route endpoints
    }
    location_ids_by_type = generate_locations(n_needed_by_type)
    all_location_ids = [lid for ids in location_ids_by_type.values() for lid in ids]
    facility_location_ids = location_ids_by_type["other"][:N_PROCESSING_FACILITIES]

    agri_product_ids = generate_products()

    # ---- Plan cross-divisional overlaps BEFORE generating VFS/Retail ----
    # (brief 5.4 / Appendix D: build overlaps in from the start)
    loyalty_customer_ids = set(random.sample(all_customer_ids, int(N_CUSTOMERS * PCT_CUSTOMERS_WITH_LOYALTY)))
    non_loyalty_ids = [c for c in all_customer_ids if c not in loyalty_customer_ids]

    wallet_customer_ids = set(random.sample(sorted(loyalty_customer_ids),
                               int(len(loyalty_customer_ids) * PCT_LOYALTY_CUSTOMERS_WITH_WALLET)))
    wallet_customer_ids |= set(random.sample(non_loyalty_ids,
                               int(len(non_loyalty_ids) * PCT_NONLOYALTY_CUSTOMERS_WITH_WALLET)))

    farmer_wallet_supplier_ids = random.sample(farmer_supplier_ids, int(len(farmer_supplier_ids) * PCT_FARMERS_WITH_WALLET))
    farmer_loan_supplier_ids = random.sample(farmer_supplier_ids, int(len(farmer_supplier_ids) * PCT_FARMERS_WITH_VFS_LOAN))
    merchant_customer_ids = random.sample(all_customer_ids, scaled(600, floor=10))

    wallet_owners = ([("customer", cid) for cid in sorted(wallet_customer_ids)] +
                     [("supplier", sid) for sid in farmer_wallet_supplier_ids])
    account_ref_ids = generate_financial_account_references(wallet_owners)

    # ---- 2. Veridian Properties ----
    property_types = generate_properties(location_ids_by_type["property"])
    tenant_kind = generate_tenants()
    n_properties = len(location_ids_by_type["property"])
    n_tenants = len(tenant_kind)
    generate_leases(n_properties, n_tenants)
    generate_property_maintenance_and_assets(n_properties)
    property_ids_warehouses = [i + 1 for i, t in enumerate(property_types) if t == "warehouse"]

    # ---- 3. Concord Logistics ----
    generate_vehicles()
    n_drivers = generate_drivers(division_of_employee)
    generate_shipments(all_location_ids)
    generate_shipment_legs(N_SHIPMENTS, N_VEHICLES, n_drivers)
    generate_routes(all_location_ids)
    generate_warehouses(location_ids_by_type["warehouse"], property_ids_warehouses)
    generate_maintenance_logs(N_VEHICLES)

    # ---- 4. Veridian Financial Services ----
    n_wallets = generate_wallet_accounts(wallet_owners, account_ref_ids)
    generate_wallet_transactions(n_wallets)
    loans_list, supplier_to_loan_ids = generate_loans(farmer_loan_supplier_ids, merchant_customer_ids)
    generate_loan_repayments(loans_list)
    kyc_customer_ids = set(wallet_customer_ids) | set(merchant_customer_ids)
    generate_kyc_records(kyc_customer_ids)
    generate_merchant_settlements()

    # ---- 5. Meridian Retail ----
    generate_stores(location_ids_by_type["store"], employee_ids_by_division)
    generate_pos_transactions(N_STORES, all_customer_ids)
    generate_transaction_line_items(N_POS_TRANSACTIONS, N_PRODUCTS)
    generate_inventory_stock_levels(N_STORES, N_PRODUCTS)
    generate_promotions(N_PRODUCTS)
    generate_loyalty_accounts(sorted(loyalty_customer_ids))
    generate_supplier_deliveries(non_farmer_delivery_supplier_ids, N_STORES)

    # ---- 6. AgriCore ----
    supplier_to_farm_ids = generate_farms(farms_plan, location_ids_by_type["farm"])
    supplier_to_farmer_id = generate_farmers(farmer_supplier_ids)
    # Simplification: any AgriCore-division employee can act as field agent
    # or quality inspector (the brief's catalog doesn't require a stricter
    # role split, and role_title on the employees table already reflects
    # who is realistically a "Field Agent" vs "Quality Inspector" for your
    # dashboards, even though this generator doesn't filter by it here).
    agricore_employee_ids = employee_ids_by_division["agricore"]
    generate_harvest_batches(supplier_to_farm_ids, agri_product_ids, agricore_employee_ids)
    n_runs = generate_processing_runs(N_HARVEST_BATCHES, facility_location_ids)
    generate_quality_grades(n_runs, agricore_employee_ids)
    generate_wholesale_shipments(n_runs, N_STORES, N_SHIPMENTS)
    loan_status_by_id = {lid: status for lid, principal, status in loans_list}
    generate_farmer_loans_reference(supplier_to_farmer_id, supplier_to_loan_ids, loan_status_by_id)

    print("\n=== Done. All CSVs written to:", os.path.abspath(OUT_DIR), "===")
    print("Next: run load_csvs.sql (see README in this folder) to bulk-load them into Supabase/Postgres.")

if __name__ == "__main__":
    main()
