// Voice, alert tone and haptics. The audio session uses .playback so the phone speaks even
// with the ring/silent switch on.

import AVFoundation
import UIKit

final class Announcer {
    private let speech = AVSpeechSynthesizer()
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var alertBuffer: AVAudioPCMBuffer?
    private var generation = 0  // bumps on every announce so a stale HELP tone never speaks late

    init() {
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? session.setActive(true)
        let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)!
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        alertBuffer = Self.makeAlert(format: format)
        try? engine.start()
        let nc = NotificationCenter.default
        // A call, Siri or an alarm interrupts audio; plugging in a projector adapter changes
        // the route. In both cases get the session and the engine running again.
        nc.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] note in
            let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            if raw == AVAudioSession.InterruptionType.ended.rawValue { self?.restartAudio() }
        }
        nc.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
            self?.restartAudio()
        }
    }

    private func restartAudio() {
        try? AVAudioSession.sharedInstance().setActive(true)
        if !engine.isRunning { try? engine.start() }
    }

    /// Silence everything now (tap-to-dismiss): a dismissed HELP must not speak afterwards.
    func stop() {
        generation += 1
        speech.stopSpeaking(at: .immediate)
        player.stop()
    }

    func announce(_ command: Command) {
        generation += 1
        let gen = generation
        speech.stopSpeaking(at: .immediate)
        player.stop()
        switch command {
        case .yes: UINotificationFeedbackGenerator().notificationOccurred(.success)
        case .no: UINotificationFeedbackGenerator().notificationOccurred(.error)
        case .help: UINotificationFeedbackGenerator().notificationOccurred(.warning)
        }
        guard command == .help, let buffer = alertBuffer else {
            say(command.spoken)
            return
        }
        if !engine.isRunning {
            do { try engine.start() } catch {
                say(command.spoken)  // no tone possible right now: still speak
                return
            }
        }
        player.scheduleBuffer(buffer, at: nil, options: [], completionCallbackType: .dataPlayedBack) { [weak self] _ in
            DispatchQueue.main.async {
                guard let self, self.generation == gen else { return }
                self.say(command.spoken)
            }
        }
        player.play()
    }

    private func say(_ text: String) {
        let u = AVSpeechUtterance(string: text)
        u.voice = AVSpeechSynthesisVoice(language: "en-US")
        u.rate = AVSpeechUtteranceDefaultSpeechRate
        u.volume = 1.0
        speech.speak(u)
    }

    /// Two-tone alert (960 / 720 Hz, 0.18 s each) repeated 3 times: about 1.1 s.
    private static func makeAlert(format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let sr = format.sampleRate
        let toneLen = Int(0.18 * sr)
        let total = toneLen * 6
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(total)),
              let data = buf.floatChannelData?[0] else { return nil }
        buf.frameLength = AVAudioFrameCount(total)
        let ramp = Int(0.01 * sr)  // 10 ms fade in/out per tone: no clicks
        for i in 0..<total {
            let k = i / toneLen
            let j = i % toneLen
            let f = k % 2 == 0 ? 960.0 : 720.0
            let env = Double(min(j, toneLen - 1 - j, ramp)) / Double(ramp)
            data[i] = Float(0.6 * min(env, 1.0) * sin(2.0 * Double.pi * f * Double(i) / sr))
        }
        return buf
    }
}
