import SwiftUI

final class AppModel: ObservableObject {
    @Published var display = DisplayModel()
    let ble = BLEClient()
    private let announcer = Announcer()
    private var testTimers: [DispatchWorkItem] = []

    init() {
        ble.onCommand = { [weak self] label in self?.handle(label) }
        ble.onDebug = { [weak self] _ in self?.display.receiveDebug(at: Date()) }
        #if DEBUG
        // Screenshot hook: `-show YES|NO|HELP` displays that result at launch (Debug builds only).
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-show"), i + 1 < args.count {
            let label = args[i + 1]
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in self?.handle(label) }
        }
        #endif
    }

    func handle(_ label: String) {
        if let cmd = display.receive(label: label, at: Date()) {
            announcer.announce(cmd)
        }
    }

    func tick() { display.tick(now: Date()) }

    func dismiss() {
        guard case .showing = display.screen else { return }
        display.dismiss()
        announcer.stop()
    }

    /// Local sound/screen check before the show; nothing goes over Bluetooth.
    func test(_ cmd: Command) { handle(cmd.rawValue) }
}

struct ContentView: View {
    @StateObject private var model = AppModel()
    @Environment(\.scenePhase) private var scenePhase
    private let ticker = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.05)) { context in
            ZStack {
                background(now: context.date)
                    .ignoresSafeArea()
                VStack(spacing: 0) {
                    statusBar(now: context.date)
                    Spacer()
                    center
                    Spacer()
                    footer
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 16)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { model.dismiss() }
        // No fade between screens: the HELP flash redraws 20x per second and kept restarting
        // a cross-fade, leaving ghosts of the Ready screen. An instant switch is also faster.
        .transaction { $0.animation = nil }
        .onReceive(ticker) { _ in model.tick() }
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                UIApplication.shared.isIdleTimerDisabled = true  // iOS may reset it in background
                model.ble.ensureConnecting()
            }
        }
        .preferredColorScheme(.dark)
        .statusBarHidden(isShowingResult)
    }

    private var isShowingResult: Bool {
        if case .showing = model.display.screen { return true }
        return false
    }

    // MARK: - Pieces

    private func background(now: Date) -> Color {
        switch model.display.screen {
        case .ready:
            return Palette.ready
        case let .showing(cmd, since):
            switch cmd {
            case .yes: return Palette.yes
            case .no: return Palette.no
            case .help:
                return DisplayModel.helpFlashBright(since: since, now: now) ? Palette.help : Palette.helpDark
            }
        }
    }

    @ViewBuilder private var center: some View {
        switch model.display.screen {
        case .ready:
            VStack(spacing: 12) {
                Text("Ready")
                    .font(.system(size: 44, weight: .semibold))
                    .foregroundStyle(Palette.readyText)
                Text("Waiting for a command")
                    .font(.system(size: 17))
                    .foregroundStyle(Palette.dimText)
            }
        case let .showing(cmd, _):
            VStack(spacing: 20) {
                Image(systemName: Palette.symbol(cmd))
                    .font(.system(size: 110, weight: .bold))
                Text(cmd.rawValue)
                    .font(.system(size: cmd == .help ? 104 : 128, weight: .heavy))
                    .minimumScaleFactor(0.5)
                    .lineLimit(1)
            }
            .foregroundStyle(.white)
            .accessibilityElement(children: .combine)
            .accessibilityLabel(cmd.spoken)
        }
    }

    private func statusBar(now: Date) -> some View {
        let (text, color) = linkText
        let active = model.ble.state == .connected && model.display.boardActive(now: now)
        return HStack(spacing: 8) {
            Circle().fill(color).frame(width: 10, height: 10)
            Text(text).font(.system(size: 15, weight: .medium))
            if active {
                Text("· receiving").font(.system(size: 15)).foregroundStyle(Palette.dimText)
            }
            Spacer()
        }
        .foregroundStyle(isShowingResult ? .white.opacity(0.85) : Palette.readyText)
        .opacity(isShowingResult ? 0 : 1)
    }

    private var linkText: (String, Color) {
        switch model.ble.state {
        case .connected: return ("Connected to SynapSys", Palette.connected)
        case .connecting: return ("Connecting…", Palette.waiting)
        case .searching: return ("Searching for SynapSys…", Palette.waiting)
        case .bluetoothOff: return ("Bluetooth is off", Palette.problem)
        case .unauthorized: return ("Allow Bluetooth in Settings", Palette.problem)
        case .unsupported: return ("Bluetooth not available on this device", Palette.problem)
        }
    }

    @ViewBuilder private var footer: some View {
        if case .ready = model.display.screen {
            HStack {
                if let last = model.display.history.first {
                    Text("Last: \(last.command.rawValue)  \(last.at.formatted(date: .omitted, time: .standard))")
                        .font(.system(size: 14))
                        .foregroundStyle(Palette.dimText)
                }
                Spacer()
                Menu {
                    Button("Test YES") { model.test(.yes) }
                    Button("Test NO") { model.test(.no) }
                    Button("Test HELP") { model.test(.help) }
                } label: {
                    Text("Test")
                        .font(.system(size: 15, weight: .medium))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 7)
                        .overlay(Capsule().stroke(Palette.dimText, lineWidth: 1))
                        .foregroundStyle(Palette.readyText)
                }
            }
        } else {
            Text("Tap to dismiss")
                .font(.system(size: 14))
                .foregroundStyle(.white.opacity(0.7))
        }
    }
}

enum Palette {
    static let ready = Color(red: 0.07, green: 0.07, blue: 0.08)
    static let readyText = Color(white: 0.88)
    static let dimText = Color(white: 0.55)
    static let yes = Color(red: 0.10, green: 0.60, blue: 0.27)
    static let no = Color(red: 0.80, green: 0.13, blue: 0.13)
    static let help = Color(red: 0.95, green: 0.50, blue: 0.02)
    static let helpDark = Color(red: 0.45, green: 0.20, blue: 0.00)
    static let connected = Color(red: 0.20, green: 0.78, blue: 0.35)
    static let waiting = Color(red: 0.95, green: 0.70, blue: 0.10)
    static let problem = Color(red: 0.90, green: 0.25, blue: 0.20)

    static func symbol(_ cmd: Command) -> String {
        switch cmd {
        case .yes: return "checkmark"
        case .no: return "xmark"
        case .help: return "exclamationmark.triangle.fill"
        }
    }
}
