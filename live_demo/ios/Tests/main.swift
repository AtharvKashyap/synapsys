// Tests for CommandLogic.swift (top-level code, so the file is main.swift). Run: ./run_logic_tests.sh

import Foundation

var failures = 0
var passed = 0

func check(_ cond: @autoclosure () -> Bool, _ name: String, line: Int = #line) {
    if cond() {
        passed += 1
    } else {
        failures += 1
        print("FAIL line \(line): \(name)")
    }
}

let t0 = Date(timeIntervalSince1970: 1_000_000)

func testLabelMapping() {
    check(Command(boardLabel: "YES") == .yes, "YES")
    check(Command(boardLabel: "NO") == .no, "NO")
    check(Command(boardLabel: "HELP") == .help, "HELP")
    check(Command(boardLabel: "F8_0") == .yes, "F8_0 -> YES")
    check(Command(boardLabel: "F12_0") == .no, "F12_0 -> NO")
    check(Command(boardLabel: "F15_0") == .help, "F15_0 -> HELP")
    check(Command(boardLabel: " yes\n") == .yes, "whitespace/case tolerant")
    check(Command(boardLabel: "REJECT") == nil, "REJECT is not a command")
    check(Command(boardLabel: "") == nil, "empty is not a command")
    check(Command(boardLabel: "F10_0") == nil, "other target is not a command")
}

func testResultHoldsThenReturnsToReady() {
    var m = DisplayModel()
    check(m.screen == .ready, "starts ready")
    check(m.receive(label: "F12_0", at: t0) == .no, "returns the command to announce")
    check(m.screen == .showing(.no, since: t0), "shows NO")
    m.tick(now: t0.addingTimeInterval(3.9))
    check(m.screen == .showing(.no, since: t0), "still showing at 3.9 s")
    m.tick(now: t0.addingTimeInterval(4.0))
    check(m.screen == .ready, "ready at 4 s")
}

func testHelpHoldsSixSeconds() {
    var m = DisplayModel()
    m.receive(label: "HELP", at: t0)
    m.tick(now: t0.addingTimeInterval(5.9))
    check(m.screen == .showing(.help, since: t0), "HELP still showing at 5.9 s")
    m.tick(now: t0.addingTimeInterval(6.0))
    check(m.screen == .ready, "HELP done at 6 s")
}

func testNewCommandReplacesCurrent() {
    var m = DisplayModel()
    m.receive(label: "YES", at: t0)
    let t1 = t0.addingTimeInterval(3)
    check(m.receive(label: "YES", at: t1) == .yes, "same command again is announced again")
    m.tick(now: t0.addingTimeInterval(4.5))
    check(m.screen == .showing(.yes, since: t1), "hold restarted by the new command")
    m.receive(label: "HELP", at: t0.addingTimeInterval(5))
    check(m.screen == .showing(.help, since: t0.addingTimeInterval(5)), "HELP replaces YES")
    check(m.history.map(\.command) == [.help, .yes, .yes], "history newest first")
}

func testIgnoredLabelChangesNothing() {
    var m = DisplayModel()
    m.receive(label: "NO", at: t0)
    check(m.receive(label: "REJECT", at: t0.addingTimeInterval(1)) == nil, "REJECT ignored")
    check(m.screen == .showing(.no, since: t0), "screen unchanged by an ignored label")
    check(m.history.count == 1, "ignored label not in history")
}

func testHistoryIsBounded() {
    var m = DisplayModel()
    for k in 0..<12 { m.receive(label: "YES", at: t0.addingTimeInterval(Double(k))) }
    check(m.history.count == DisplayModel.historyLimit, "history capped")
}

func testDismiss() {
    var m = DisplayModel()
    m.receive(label: "NO", at: t0)
    m.dismiss()
    check(m.screen == .ready, "tap dismisses")
}

func testBoardActivityExpires() {
    var m = DisplayModel()
    check(!m.boardActive(now: t0), "inactive before any debug")
    m.receiveDebug(at: t0)
    check(m.boardActive(now: t0.addingTimeInterval(1.4)), "active 1.4 s after debug")
    check(!m.boardActive(now: t0.addingTimeInterval(1.5)), "inactive 1.5 s after debug")
}

func testHelpFlashIs2HzForThreeSeconds() {
    let on = (0..<12).map { DisplayModel.helpFlashBright(since: t0, now: t0.addingTimeInterval(Double($0) * 0.25)) }
    check(on == [true, false, true, false, true, false, true, false, true, false, true, false],
          "alternates every 0.25 s")
    check(DisplayModel.helpFlashBright(since: t0, now: t0.addingTimeInterval(3.1)), "steady after 3 s")
    check(DisplayModel.helpFlashBright(since: t0, now: t0.addingTimeInterval(3.3)), "steady after 3 s (2)")
}

testLabelMapping()
testResultHoldsThenReturnsToReady()
testHelpHoldsSixSeconds()
testNewCommandReplacesCurrent()
testIgnoredLabelChangesNothing()
testHistoryIsBounded()
testDismiss()
testBoardActivityExpires()
testHelpFlashIs2HzForThreeSeconds()
print(failures == 0 ? "PASS (\(passed) checks)" : "FAIL: \(failures) of \(passed + failures) checks")
exit(failures == 0 ? 0 : 1)
