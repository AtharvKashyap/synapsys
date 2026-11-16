// Laptop Bluetooth relay (demo fallback for a board whose radio browns out).
//
// Advertises as "SynapSys" with the board's GATT service (firmware/esp32/main/ble_output.c),
// so the iPhone app connects to the Mac unchanged. It decides nothing: the stage app writes the
// board's own lines to stdin and this forwards them.
//
// stdin, one line each:  C <board label>   -> Command characteristic (read + notify)
//                        D <text>          -> Debug characteristic (notify)
//                        S <text>          -> Status characteristic (read)
// stdout, one line each: state <poweredOn|...> | advertising | error <text> |
//                        phone <n>  (centrals subscribed to Command) | sent C <label> <n>
// Exits when stdin closes (the stage app quit).

import CoreBluetooth
import Foundation

let base = "-2C7E-4B8A-9D61-53594E415053"
let serviceUUID = CBUUID(string: "A5F30001" + base)
let commandUUID = CBUUID(string: "A5F30002" + base)
let debugUUID = CBUUID(string: "A5F30003" + base)
let statusUUID = CBUUID(string: "A5F30004" + base)

func say(_ s: String) {
    FileHandle.standardOutput.write((s + "\n").data(using: .utf8)!)
}

final class Relay: NSObject, CBPeripheralManagerDelegate {
    private let queue = DispatchQueue(label: "relay")
    private var pm: CBPeripheralManager!
    private let command = CBMutableCharacteristic(
        type: commandUUID, properties: [.read, .notify], value: nil, permissions: [.readable])
    private let debug = CBMutableCharacteristic(
        type: debugUUID, properties: [.notify], value: nil, permissions: [.readable])
    private let status = CBMutableCharacteristic(
        type: statusUUID, properties: [.read], value: nil, permissions: [.readable])
    private var values: [CBUUID: Data] = [:]
    private var subscribers: [UUID: CBCentral] = [:]  // to Command
    private var pending: [(CBMutableCharacteristic, Data)] = []

    override init() {
        super.init()
        values[commandUUID] = Data()
        values[statusUUID] = "relay: laptop forwards the ESP32-S3's decisions".data(using: .utf8)!
        pm = CBPeripheralManager(delegate: self, queue: queue)
    }

    // Called from the stdin thread.
    func input(_ line: String) {
        queue.async { [self] in
            guard line.count >= 2 else { return }
            let kind = line.first!
            let text = String(line.dropFirst(2))
            let data = text.data(using: .utf8)!
            switch kind {
            case "C":
                values[commandUUID] = data
                push(command, data)
                say("sent C \(text) \(subscribers.count)")
            case "D":
                push(debug, data)
            case "S":
                values[statusUUID] = data
            default:
                break
            }
        }
    }

    private func push(_ ch: CBMutableCharacteristic, _ data: Data) {
        guard pm.state == .poweredOn else { return }
        if !pending.isEmpty || !pm.updateValue(data, for: ch, onSubscribedCentrals: nil) {
            if ch === debug { pending.removeAll { $0.0 === debug } }  // only the latest debug
            pending.append((ch, data))  // sent from peripheralManagerIsReady
        }
    }

    func peripheralManagerIsReady(toUpdateSubscribers peripheral: CBPeripheralManager) {
        while let (ch, data) = pending.first {
            if !pm.updateValue(data, for: ch, onSubscribedCentrals: nil) { return }
            pending.removeFirst()
        }
    }

    func peripheralManagerDidUpdateState(_ peripheral: CBPeripheralManager) {
        let names: [CBManagerState: String] = [
            .poweredOn: "poweredOn", .poweredOff: "poweredOff", .unauthorized: "unauthorized",
            .unsupported: "unsupported", .resetting: "resetting", .unknown: "unknown",
        ]
        say("state \(names[peripheral.state] ?? "?")")
        subscribers.removeAll()
        pending.removeAll()
        guard peripheral.state == .poweredOn else { return }
        let svc = CBMutableService(type: serviceUUID, primary: true)
        svc.characteristics = [command, debug, status]
        peripheral.removeAllServices()
        peripheral.add(svc)
    }

    func peripheralManager(_ peripheral: CBPeripheralManager, didAdd service: CBService, error: Error?) {
        if let error { say("error add service: \(error.localizedDescription)"); return }
        peripheral.startAdvertising([
            CBAdvertisementDataLocalNameKey: "SynapSys",
            CBAdvertisementDataServiceUUIDsKey: [serviceUUID],
        ])
    }

    func peripheralManagerDidStartAdvertising(_ peripheral: CBPeripheralManager, error: Error?) {
        say(error.map { "error advertising: \($0.localizedDescription)" } ?? "advertising")
    }

    func peripheralManager(_ peripheral: CBPeripheralManager, didReceiveRead request: CBATTRequest) {
        let v = values[request.characteristic.uuid] ?? Data()
        guard request.offset <= v.count else {
            peripheral.respond(to: request, withResult: .invalidOffset)
            return
        }
        request.value = v.subdata(in: request.offset..<v.count)
        peripheral.respond(to: request, withResult: .success)
    }

    func peripheralManager(_ peripheral: CBPeripheralManager, central: CBCentral,
                           didSubscribeTo characteristic: CBCharacteristic) {
        guard characteristic.uuid == commandUUID else { return }
        subscribers[central.identifier] = central
        say("phone \(subscribers.count)")
    }

    func peripheralManager(_ peripheral: CBPeripheralManager, central: CBCentral,
                           didUnsubscribeFrom characteristic: CBCharacteristic) {
        guard characteristic.uuid == commandUUID else { return }
        subscribers[central.identifier] = nil
        say("phone \(subscribers.count)")
    }
}

setvbuf(stdout, nil, _IOLBF, 0)
let relay = Relay()
Thread.detachNewThread {
    while let line = readLine() {
        relay.input(line.trimmingCharacters(in: .whitespacesAndNewlines))
    }
    exit(0)  // stage app closed our stdin
}
dispatchMain()
