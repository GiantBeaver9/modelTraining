"""Compile + run every branchless C/Rust solution, keyword-check it's branchless, and emit a verified
code bank (sovereign/data/code_bank.json) for the dataset generator. Anything that fails to compile,
fails its test, or trips the branchless linter is DROPPED with a report — so every gold code example
that reaches training is proven correct AND free of if/for/while.
"""
import json, subprocess, sys, tempfile, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from branchless_lint import lint_snippet  # noqa: E402

# Each entry: id, task (canonical NL phrase), lang, code (the fenced body), test (full compilable program).
BANK = []

def C(id, task, code, test): BANK.append(dict(id=id, task=task, lang="c", code=code, test=test))
def R(id, task, code, test): BANK.append(dict(id=id, task=task, lang="rust", code=code, test=test))

# ---------------------------------------------------------------- C (ternary + recursion) ----------
C("fact_c", "compute the factorial of n",
  "unsigned long fact(unsigned long n) {\n    return n == 0 ? 1 : n * fact(n - 1);\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(fact(0)==1); assert(fact(5)==120); assert(fact(7)==5040); return 0; }")

C("fib_c", "compute the nth Fibonacci number",
  "unsigned long fib(unsigned n) {\n    return n < 2 ? n : fib(n - 1) + fib(n - 2);\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(fib(0)==0); assert(fib(1)==1); assert(fib(10)==55); assert(fib(15)==610); return 0; }")

C("gcd_c", "compute the greatest common divisor of two integers",
  "unsigned long gcd(unsigned long a, unsigned long b) {\n    return b == 0 ? a : gcd(b, a % b);\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(gcd(12,8)==4); assert(gcd(7,0)==7); assert(gcd(48,36)==12); return 0; }")

C("abs_c", "compute the absolute value of an integer",
  "long iabs(long x) {\n    return x < 0 ? -x : x;\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(iabs(-5)==5); assert(iabs(5)==5); assert(iabs(0)==0); return 0; }")

C("sign_c", "return the sign of a number as -1, 0, or 1",
  "int sign(long x) {\n    return (x > 0) - (x < 0);\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(sign(-9)==-1); assert(sign(0)==0); assert(sign(4)==1); return 0; }")

C("max2_c", "return the larger of two integers",
  "long max2(long a, long b) {\n    return a > b ? a : b;\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(max2(3,7)==7); assert(max2(9,2)==9); assert(max2(4,4)==4); return 0; }")

C("min2_c", "return the smaller of two integers",
  "long min2(long a, long b) {\n    return a < b ? a : b;\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(min2(3,7)==3); assert(min2(9,2)==2); return 0; }")

C("pow_c", "compute base raised to a non-negative integer power",
  "unsigned long ipow(unsigned long b, unsigned e) {\n    return e == 0 ? 1 : b * ipow(b, e - 1);\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(ipow(2,0)==1); assert(ipow(2,10)==1024); assert(ipow(5,3)==125); return 0; }")

C("iseven_c", "check whether an integer is even",
  "int is_even(long n) {\n    return n % 2 == 0;\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(is_even(4)); assert(!is_even(7)); assert(is_even(0)); return 0; }")

C("sumn_c", "sum the integers from 1 to n",
  "unsigned long sum_to(unsigned n) {\n    return n == 0 ? 0 : n + sum_to(n - 1);\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(sum_to(0)==0); assert(sum_to(5)==15); assert(sum_to(100)==5050); return 0; }")

C("sumarr_c", "sum the elements of an integer array",
  "long sum(const long *a, int n) {\n    return n == 0 ? 0 : a[n - 1] + sum(a, n - 1);\n}",
  "#include <assert.h>\n{CODE}\nint main(){ long a[]={1,2,3,4}; assert(sum(a,4)==10); assert(sum(a,0)==0); return 0; }")

C("count_c", "count how many times a value appears in an array",
  "int count(const int *a, int n, int v) {\n    return n == 0 ? 0 : (a[n - 1] == v) + count(a, n - 1, v);\n}",
  "#include <assert.h>\n{CODE}\nint main(){ int a[]={2,2,3,2}; assert(count(a,4,2)==3); assert(count(a,4,9)==0); return 0; }")

C("digitsum_c", "sum the decimal digits of a non-negative integer",
  "int digit_sum(unsigned long n) {\n    return n == 0 ? 0 : (int)(n % 10) + digit_sum(n / 10);\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(digit_sum(0)==0); assert(digit_sum(123)==6); assert(digit_sum(9999)==36); return 0; }")

C("strlen_c", "compute the length of a C string",
  "unsigned slen(const char *s) {\n    return *s == 0 ? 0 : 1 + slen(s + 1);\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(slen(\"\")==0); assert(slen(\"hello\")==5); return 0; }")

C("reverse_c", "reverse a string in place",
  "#include <string.h>\nstatic void rev(char *s, int i, int j) {\n    char t;\n    (i < j) && ((t = s[i], s[i] = s[j], s[j] = t), rev(s, i + 1, j - 1), 0);\n}\nvoid reverse(char *s) {\n    rev(s, 0, (int)strlen(s) - 1);\n}",
  "#include <assert.h>\n#include <string.h>\n{CODE}\nint main(){ char a[]=\"abc\"; reverse(a); assert(strcmp(a,\"cba\")==0); char b[]=\"\"; reverse(b); assert(strcmp(b,\"\")==0); char c[]=\"racecar\"; reverse(c); assert(strcmp(c,\"racecar\")==0); return 0; }")

C("palindrome_c", "check whether a string is a palindrome",
  "#include <string.h>\nstatic int pal(const char *s, int i, int j) {\n    return i >= j ? 1 : (s[i] == s[j] && pal(s, i + 1, j - 1));\n}\nint is_palindrome(const char *s) {\n    return pal(s, 0, (int)strlen(s) - 1);\n}",
  "#include <assert.h>\n#include <string.h>\n{CODE}\nint main(){ assert(is_palindrome(\"racecar\")); assert(is_palindrome(\"\")); assert(!is_palindrome(\"abc\")); assert(is_palindrome(\"abba\")); return 0; }")

C("clamp_c", "clamp a value between a low and high bound",
  "long clamp(long x, long lo, long hi) {\n    return x < lo ? lo : (x > hi ? hi : x);\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(clamp(5,0,10)==5); assert(clamp(-3,0,10)==0); assert(clamp(99,0,10)==10); return 0; }")

C("c2f_c", "convert Celsius to Fahrenheit",
  "double c_to_f(double c) {\n    return c * 9.0 / 5.0 + 32.0;\n}",
  "#include <assert.h>\n{CODE}\nint main(){ assert(c_to_f(0)==32.0); assert(c_to_f(100)==212.0); return 0; }")

# ---------------------------------------------------------------- Rust (match + iterators) ----------
R("fact_rs", "compute the factorial of n",
  "fn fact(n: u64) -> u64 {\n    match n { 0 => 1, _ => n * fact(n - 1) }\n}",
  "{CODE}\nfn main(){ assert_eq!(fact(0),1); assert_eq!(fact(5),120); assert_eq!(fact(7),5040); }")

R("fib_rs", "compute the nth Fibonacci number",
  "fn fib(n: u64) -> u64 {\n    match n { 0 => 0, 1 => 1, _ => fib(n - 1) + fib(n - 2) }\n}",
  "{CODE}\nfn main(){ assert_eq!(fib(0),0); assert_eq!(fib(1),1); assert_eq!(fib(10),55); assert_eq!(fib(15),610); }")

R("fib_fast_rs", "compute the nth Fibonacci number efficiently (linear, tail-recursive)",
  "fn fib(n: u64) -> u64 {\n    fn go(n: u64, a: u64, b: u64) -> u64 {\n        match n { 0 => a, _ => go(n - 1, b, a + b) }\n    }\n    go(n, 0, 1)\n}",
  "{CODE}\nfn main(){ assert_eq!(fib(0),0); assert_eq!(fib(10),55); assert_eq!(fib(30),832040); }")

R("gcd_rs", "compute the greatest common divisor of two integers",
  "fn gcd(a: u64, b: u64) -> u64 {\n    match b { 0 => a, _ => gcd(b, a % b) }\n}",
  "{CODE}\nfn main(){ assert_eq!(gcd(12,8),4); assert_eq!(gcd(7,0),7); assert_eq!(gcd(48,36),12); }")

R("sign_rs", "return the sign of a number as -1, 0, or 1",
  "fn sign(x: i64) -> i64 {\n    (x > 0) as i64 - (x < 0) as i64\n}",
  "{CODE}\nfn main(){ assert_eq!(sign(-9),-1); assert_eq!(sign(0),0); assert_eq!(sign(4),1); }")

R("abs_rs", "compute the absolute value of an integer",
  "fn iabs(x: i64) -> i64 {\n    x * ((x > 0) as i64 - (x < 0) as i64)\n}",
  "{CODE}\nfn main(){ assert_eq!(iabs(-5),5); assert_eq!(iabs(5),5); assert_eq!(iabs(0),0); }")

R("max2_rs", "return the larger of two integers",
  "fn max2(a: i64, b: i64) -> i64 {\n    match a > b { true => a, false => b }\n}",
  "{CODE}\nfn main(){ assert_eq!(max2(3,7),7); assert_eq!(max2(9,2),9); }")

R("pow_rs", "compute base raised to a non-negative integer power",
  "fn ipow(b: u64, e: u32) -> u64 {\n    match e { 0 => 1, _ => b * ipow(b, e - 1) }\n}",
  "{CODE}\nfn main(){ assert_eq!(ipow(2,0),1); assert_eq!(ipow(2,10),1024); assert_eq!(ipow(5,3),125); }")

R("sumslice_rs", "sum the elements of a slice",
  "fn total(xs: &[i64]) -> i64 {\n    xs.iter().sum()\n}",
  "{CODE}\nfn main(){ assert_eq!(total(&[1,2,3,4]),10); assert_eq!(total(&[]),0); }")

R("sumrec_rs", "sum the elements of a slice with recursion",
  "fn total(xs: &[i64]) -> i64 {\n    match xs.split_first() {\n        None => 0,\n        Some((h, rest)) => h + total(rest),\n    }\n}",
  "{CODE}\nfn main(){ assert_eq!(total(&[1,2,3,4]),10); assert_eq!(total(&[]),0); assert_eq!(total(&[-1,1]),0); }")

R("doubled_rs", "double every element of a slice",
  "fn doubled(xs: &[i32]) -> Vec<i32> {\n    xs.iter().map(|x| x * 2).collect()\n}",
  "{CODE}\nfn main(){ assert_eq!(doubled(&[1,2,3]), vec![2,4,6]); assert_eq!(doubled(&[]), Vec::<i32>::new()); }")

R("evens_rs", "keep only the even numbers of a slice",
  "fn evens(xs: &[i32]) -> Vec<i32> {\n    xs.iter().copied().filter(|x| x % 2 == 0).collect()\n}",
  "{CODE}\nfn main(){ assert_eq!(evens(&[1,2,3,4]), vec![2,4]); assert_eq!(evens(&[1,3]), Vec::<i32>::new()); }")

R("maxslice_rs", "find the maximum of a non-empty slice",
  "fn maxv(xs: &[i64]) -> i64 {\n    *xs.iter().max().unwrap()\n}",
  "{CODE}\nfn main(){ assert_eq!(maxv(&[1,5,2]),5); assert_eq!(maxv(&[-3,-1,-7]),-1); }")

R("reverse_rs", "reverse a string",
  "fn reverse(s: &str) -> String {\n    s.chars().rev().collect()\n}",
  "{CODE}\nfn main(){ assert_eq!(reverse(\"abc\"), \"cba\"); assert_eq!(reverse(\"\"), \"\"); assert_eq!(reverse(\"racecar\"), \"racecar\"); }")

R("iseven_rs", "check whether an integer is even",
  "fn is_even(n: i64) -> bool {\n    n % 2 == 0\n}",
  "{CODE}\nfn main(){ assert!(is_even(4)); assert!(!is_even(7)); assert!(is_even(0)); }")

R("count_rs", "count how many times a value appears in a slice",
  "fn count(xs: &[i32], v: i32) -> usize {\n    xs.iter().filter(|&&x| x == v).count()\n}",
  "{CODE}\nfn main(){ assert_eq!(count(&[2,2,3,2],2),3); assert_eq!(count(&[2,2,3,2],9),0); }")

R("palindrome_rs", "check whether a string is a palindrome",
  "fn is_palindrome(s: &str) -> bool {\n    let c: Vec<char> = s.chars().collect();\n    c.iter().eq(c.iter().rev())\n}",
  "{CODE}\nfn main(){ assert!(is_palindrome(\"racecar\")); assert!(is_palindrome(\"\")); assert!(!is_palindrome(\"abc\")); }")

R("clamp_rs", "clamp a value between a low and high bound",
  "fn clamp(x: i64, lo: i64, hi: i64) -> i64 {\n    x.max(lo).min(hi)\n}",
  "{CODE}\nfn main(){ assert_eq!(clamp(5,0,10),5); assert_eq!(clamp(-3,0,10),0); assert_eq!(clamp(99,0,10),10); }")

R("digitsum_rs", "sum the decimal digits of a non-negative integer",
  "fn digit_sum(n: u64) -> u64 {\n    match n { 0 => 0, _ => n % 10 + digit_sum(n / 10) }\n}",
  "{CODE}\nfn main(){ assert_eq!(digit_sum(0),0); assert_eq!(digit_sum(123),6); assert_eq!(digit_sum(9999),36); }")

R("sumto_rs", "sum the integers from 1 to n",
  "fn sum_to(n: u64) -> u64 {\n    (1..=n).sum()\n}",
  "{CODE}\nfn main(){ assert_eq!(sum_to(5),15); assert_eq!(sum_to(100),5050); }")

# ---------------------------------------------------------------- compile + run ----------
def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=90, **kw)

def verify(entry, tmp):
    code, test = entry["code"], entry["test"].replace("{CODE}", entry["code"])
    # branchless keyword gate on the SHOWN code (what the model emits)
    bl = lint_snippet(code)
    if not bl.ok:
        return False, f"branchless-fail: {bl.hits}"
    if entry["lang"] == "c":
        src = Path(tmp) / f"{entry['id']}.c"; src.write_text(test)
        exe = Path(tmp) / entry["id"]
        c = run(["gcc", "-O2", "-w", str(src), "-o", str(exe)])
        if c.returncode != 0:
            return False, "gcc: " + c.stderr.strip()[:200]
    else:
        src = Path(tmp) / f"{entry['id']}.rs"; src.write_text(test)
        exe = Path(tmp) / entry["id"]
        c = run(["rustc", "-O", "-A", "warnings", str(src), "-o", str(exe)])
        if c.returncode != 0:
            return False, "rustc: " + c.stderr.strip()[:200]
    r = run([str(exe)])
    if r.returncode != 0:
        return False, f"test failed (rc={r.returncode}): {r.stderr.strip()[:160]}"
    return True, "ok"

def main():
    verified, failures = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for e in BANK:
            ok, msg = verify(e, tmp)
            print(f"  {'OK ' if ok else 'XX '} {e['lang']:<4} {e['id']:<16} {msg if not ok else ''}")
            (verified if ok else failures).append(e if ok else (e['id'], msg))
    print(f"\nverified {len(verified)}/{len(BANK)}  (c={sum(v['lang']=='c' for v in verified)}, "
          f"rust={sum(v['lang']=='rust' for v in verified)})")
    if failures:
        print("FAILURES:", [f for f in failures])
    out = REPO / "sovereign" / "data" / "code_bank.json"
    out.write_text(json.dumps(verified, indent=1))
    print(f"wrote {out}")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
