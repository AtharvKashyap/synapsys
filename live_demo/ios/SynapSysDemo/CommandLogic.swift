// Pure display logic (Foundation only, so it is unit-tested on the Mac with swiftc:
// see ../Tests/run_logic_tests.sh). No UIKit, no Bluetooth, no audio here.

import Foundation

enum Command: String, Equatable {
    case yes = "YES"
    case no = "NO"
    case help = "HELP"

    /// Board label -> command. The demo firmware runs the Wang benchmark config, whose labels
    /// are F8_0 / F12_0 / F15_0 (8 Hz stands in for YES); the real SynapSys firmware sends
    /// YES / NO / HELP. Anything else (REJECT, empty, other targets) is not a command.
    init?(boardLabel: String) {
        switch boardLabel.trimmingCharacters(in: .whitespacesAndNewlines).uppercased() {
        case "YES", "F8_0": self = .yes
        case "NO", "F12_0": self = .no
        case "HELP", "F15_0": self = .help
        default: return nil
        }
    }

    /// How long the result stays on screen before returning to Ready.
    var holdSeconds: TimeInterval { self == .help ? 6 : 4 }

    /// What the phone says out loud.
    var spoken: String {
        switch self {
        case .yes: return "Yes"
        case .no: return "No"
        case .help: return "Help"
        }
    }
}

enum Screen: Equatable {
    case ready
    case showing(Command, since: Date)
}

struct DisplayModel {
    /// Debug notifications arrive every 0.5 s while the board is decoding.
    static let activityWindow: TimeInterval = 1.5
    static let historyLimit = 5

    private(set) var screen: Screen = .ready
    private(set) var lastDebugAt: Date?
    private(set) var history: [(command: Command, at: Date)] = []

    /// A Command notification. Returns the command to announce, or nil if the label is not one.
    /// A new command always replaces whatever is showing (and restarts its hold time).
    @discardableResult
    mutating func receive(label: String, at now: Date) -> Command? {
        guard let cmd = Command(boardLabel: label) else { return nil }
        screen = .showing(cmd, since: now)
        history.insert((cmd, now), at: 0)
        if history.count > Self.historyLimit { history.removeLast() }
        return cmd
    }

    mutating func receiveDebug(at now: Date) {
        lastDebugAt = now
    }

    /// Call regularly (e.g. 10x per second): returns to Ready once the hold time is over.
    mutating func tick(now: Date) {
        if case let .showing(cmd, since) = screen, now.timeIntervalSince(since) >= cmd.holdSeconds {
            screen = .ready
        }
    }

    mutating func dismiss() {
        screen = .ready
    }

    /// True while the board is streaming decisions (the "thinking" dot on the Ready screen).
    func boardActive(now: Date) -> Bool {
        guard let t = lastDebugAt else { return false }
        return now.timeIntervalSince(t) < Self.activityWindow
    }

    /// For HELP: true on the bright half of a 2 Hz flash during the first 3 s (2 Hz is below
    /// the 3 Hz photosensitivity limit, and the phone may be mirrored onto a projector).
    static func helpFlashBright(since: Date, now: Date) -> Bool {
        let t = now.timeIntervalSince(since)
        guard t < 3 else { return true }
        return Int(t * 4) % 2 == 0  // 0.25 s bright, 0.25 s dark
    }
}
