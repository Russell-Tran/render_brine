// A minimal test harness. The Xcode command-line tools don't ship XCTest, so
// tests are plain functions registered with `test(...)`, checked with
// `expect`/`expectEqual`, and `finish()` exits non-zero if anything failed.

import Foundation

private var passed = 0
private var failed: [String] = []
private var currentFailures: [String] = []

func test(_ name: String, _ body: () throws -> Void) {
    currentFailures = []
    do {
        try body()
    } catch {
        currentFailures.append("threw \(error)")
    }
    if currentFailures.isEmpty {
        passed += 1
        print("  ok    \(name)")
    } else {
        failed.append(name)
        print("  FAIL  \(name)")
        for failure in currentFailures { print("        \(failure)") }
    }
}

func expect(_ condition: Bool, _ message: @autoclosure () -> String = "",
            file: StaticString = #fileID, line: UInt = #line) {
    if !condition {
        currentFailures.append("\(file):\(line) expectation failed \(message())")
    }
}

func expectEqual<T: Equatable>(_ actual: T, _ expected: T,
                               file: StaticString = #fileID, line: UInt = #line) {
    if actual != expected {
        currentFailures.append("\(file):\(line) expected \(expected), got \(actual)")
    }
}

func expectThrows(_ body: () throws -> Void, file: StaticString = #fileID, line: UInt = #line) {
    do {
        try body()
        currentFailures.append("\(file):\(line) expected an error, none was thrown")
    } catch {}
}

func section(_ title: String) {
    print("\n\(title)")
}

func finish() -> Never {
    print("\n\(passed) passed, \(failed.count) failed")
    exit(failed.isEmpty ? 0 : 1)
}
