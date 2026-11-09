"""
WAL Simulation: Two transactions on a bank database.
  T1: Alice -> Bob $200   (COMMITS successfully)
  T2: Bob -> Charlie $100  (CRASHES before commit)
"""

# ============================================================
# STABLE STORAGE (survives crashes)
# ============================================================
disk = {"Alice": 1000, "Bob": 500, "Charlie": 0}   # the actual database
log  = []                                           # the write-ahead log

# ============================================================
# VOLATILE STORAGE (lost on crash)
# ============================================================
buffer = {}   # in-memory changes not yet flushed to disk

# ============================================================
# PRIMITIVES
# ============================================================

def log_write(record):
    """Append a record to the WAL on stable storage."""
    log.append(record)
    print(f"    📝 LOG: {record}")

def read(key):
    """Read: buffer first, then disk."""
    if key in buffer:
        val = buffer[key]
        print(f"    📖 READ {key} = {val} (from buffer)")
        return val
    val = disk[key]
    print(f"    📖 READ {key} = {val} (from disk)")
    return val

def update(txn, key, old_val, new_val):
    """WAL protocol: log FIRST, then update buffer."""
    log_write({"type": "UPDATE", "txn": txn, "key": key,
               "old": old_val, "new": new_val})
    buffer[key] = new_val
    print(f"    ✏️  BUFFER {key} = {new_val}  (volatile)")

def flush_buffer():
    """Simulate buffer manager writing pages to disk."""
    for key, val in buffer.items():
        disk[key] = val
        print(f"    💾 DISK  {key} = {val}  (flushed)")
    buffer.clear()

def commit(txn):
    """Commit: log COMMIT record, then flush buffer."""
    log_write({"type": "COMMIT", "txn": txn})
    flush_buffer()

def crash():
    """Crash: volatile buffer is lost forever."""
    global buffer
    buffer.clear()
    print(f"\n  💥 CRASH! Volatile buffer lost. Disk & log survive.")

# ============================================================
# SCENARIO BEGINS
# ============================================================

print("=" * 62)
print("INITIAL STATE")
print("=" * 62)
print(f"  Disk:   {disk}")
print(f"  Log:    {log}")
print(f"  Buffer: {buffer}")

# ----- T1: Alice -> Bob $200 (COMMITS) -----
print("\n" + "=" * 62)
print("T1: Transfer $200 from Alice to Bob  (COMMITS)")
print("=" * 62)

a = read("Alice")                     # 1000
update("T1", "Alice", a, a - 200)     # log: Alice 1000->800, buffer: 800

b = read("Bob")                       # 500
update("T1", "Bob", b, b + 200)       # log: Bob 500->700, buffer: 700

commit("T1")                          # log COMMIT T1, flush buffer to disk

print(f"\n  Disk after T1: {disk}")

# ----- T2: Bob -> Charlie $100 (CRASHES) -----
print("\n" + "=" * 62)
print("T2: Transfer $100 from Bob to Charlie  (CRASHES)")
print("=" * 62)

b = read("Bob")                       # 700 (from disk, T1 committed)
update("T2", "Bob", b, b - 100)       # log: Bob 700->600, buffer: 600
flush_buffer()                        # buffer manager flushes Bob to disk

c = read("Charlie")                   # 0
update("T2", "Charlie", c, c + 100)   # log: Charlie 0->100, buffer: 100
flush_buffer()                        # buffer manager flushes Charlie to disk

# CRASH happens here -- before T2 commits!
print("\n  ... about to commit T2 ...")
crash()

print(f"\n  Disk after crash: {disk}")
print(f"  Log after crash:  {log}")
print(f"  Buffer (lost):    {buffer}")

# ============================================================
# RECOVERY
# ============================================================
print("\n" + "=" * 62)
print("RECOVERY: Reading the log")
print("=" * 62)

# --- Step 1: Identify committed vs. uncommitted transactions ---
committed = set()
all_txns = set()
for rec in log:
    if rec["type"] == "COMMIT":
        committed.add(rec["txn"])
    elif rec["type"] == "UPDATE":
        all_txns.add(rec["txn"])
uncommitted = all_txns - committed

print(f"\n  Committed:   {committed}")
print(f"  Uncommitted: {uncommitted}")

# --- Step 2: REDO committed transactions (re-apply new values) ---
print("\n  --- REDO (re-apply changes of committed transactions) ---")
for rec in log:
    if rec["type"] == "UPDATE" and rec["txn"] in committed:
        key, new_val = rec["key"], rec["new"]
        if disk[key] != new_val:
            disk[key] = new_val
            print(f"    🔄 REDO {key} = {new_val}  (re-applied)")
        else:
            print(f"    ✓  {key} = {new_val}  (already correct)")

# --- Step 3: UNDO uncommitted transactions (restore old values) ---
print("\n  --- UNDO (roll back uncommitted transactions) ---")
for rec in reversed(log):
    if rec["type"] == "UPDATE" and rec["txn"] in uncommitted:
        key, old_val = rec["key"], rec["old"]
        if disk[key] != old_val:
            disk[key] = old_val
            print(f"    ⏪ UNDO {key} = {old_val}  (restored)")
        else:
            print(f"    ✓  {key} = {old_val}  (already correct)")

# --- Final state ---
print("\n" + "=" * 62)
print("FINAL STATE (after recovery)")
print("=" * 62)
print(f"  Disk:   {disk}")
print(f"  Log:    {log}")
print(f"  Buffer: {buffer}")
print("\n✅ Correct outcome:")
print("   T1 committed → Alice=800, Bob=700  (preserved)")
print("   T2 crashed  → Bob=700, Charlie=0   (undone)")