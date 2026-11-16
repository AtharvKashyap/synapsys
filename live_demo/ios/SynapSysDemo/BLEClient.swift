// CoreBluetooth central for the SynapSys board (GATT server in firmware/esp32/main/ble_output.c).
// Finds the board by name or service UUID, subscribes to Command + Debug, and reconnects by
// itself whenever the link drops.

import CoreBluetooth
import Foundation

enum LinkState: Equatable {
    case bluetoothOff
    case unauthorized
    case unsupported
    case searching
    case connecting
    case connected
}

final class BLEClient: NSObject, ObservableObject {
    static let service = CBUUID(string: "A5F30001-2C7E-4B8A-9D61-53594E415053")
    static let command = CBUUID(string: "A5F30002-2C7E-4B8A-9D61-53594E415053")
    static let debug = CBUUID(string: "A5F30003-2C7E-4B8A-9D61-53594E415053")
    static let deviceName = "SynapSys"

    @Published private(set) var state: LinkState = .searching

    /// Called on the main queue with the raw Command / Debug text.
    var onCommand: ((String) -> Void)?
    var onDebug: ((String) -> Void)?

    private var central: CBCentralManager!
    private var board: CBPeripheral?
    private var watchdog: DispatchWorkItem?

    override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: .main)
    }

    /// Safe to call any time (e.g. when the app comes back to the foreground).
    func ensureConnecting() {
        guard central.state == .poweredOn else { return }
        if let p = board, p.state == .connected || p.state == .connecting { return }
        startScan()
    }

    private func startScan() {
        guard central.state == .poweredOn, !central.isScanning else { return }
        state = .searching
        // No service filter: the 128-bit UUID is only in the scan response, so match by name
        // or UUID ourselves.
        central.scanForPeripherals(withServices: nil, options: [CBCentralManagerScanOptionAllowDuplicatesKey: false])
    }

    /// The link to `peripheral` is gone: forget it and look again shortly. Callbacks about an
    /// older peripheral (after a reconnect already happened) are ignored.
    private func lost(_ peripheral: CBPeripheral) {
        guard peripheral === board else { return }
        watchdog?.cancel()
        board = nil
        state = .searching
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in self?.ensureConnecting() }
    }

    /// Give up on a connection attempt that has not finished setting up after 8 s (the board
    /// stops advertising while connected, so a stuck attempt would never recover by itself).
    private func armWatchdog(for peripheral: CBPeripheral) {
        watchdog?.cancel()
        let item = DispatchWorkItem { [weak self] in
            guard let self, peripheral === self.board, self.state != .connected else { return }
            self.central.cancelPeripheralConnection(peripheral)
            self.lost(peripheral)
        }
        watchdog = item
        DispatchQueue.main.asyncAfter(deadline: .now() + 8.0, execute: item)
    }

    private func fail(_ peripheral: CBPeripheral) {
        central.cancelPeripheralConnection(peripheral)  // didDisconnect -> lost() -> retry
    }
}

extension BLEClient: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        if central.state != .poweredOn {
            watchdog?.cancel()
            board = nil  // CoreBluetooth drops every connection when Bluetooth goes off
        }
        switch central.state {
        case .poweredOn: startScan()
        case .unauthorized: state = .unauthorized
        case .unsupported: state = .unsupported
        case .unknown, .resetting: state = .searching  // transient at launch
        default: state = .bluetoothOff
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                        advertisementData: [String: Any], rssi RSSI: NSNumber) {
        let name = (advertisementData[CBAdvertisementDataLocalNameKey] as? String) ?? peripheral.name
        let uuids = (advertisementData[CBAdvertisementDataServiceUUIDsKey] as? [CBUUID]) ?? []
        guard name == Self.deviceName || uuids.contains(Self.service) else { return }
        central.stopScan()
        board = peripheral
        peripheral.delegate = self
        state = .connecting
        central.connect(peripheral)
        armWatchdog(for: peripheral)
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        peripheral.discoverServices([Self.service])
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        lost(peripheral)
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral,
                        error: Error?) {
        lost(peripheral)
    }
}

extension BLEClient: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard error == nil, let svc = peripheral.services?.first(where: { $0.uuid == Self.service }) else {
            fail(peripheral)  // not our board, or discovery failed
            return
        }
        peripheral.discoverCharacteristics([Self.command, Self.debug], for: svc)
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        let chars = (service.characteristics ?? []).filter { $0.uuid == Self.command || $0.uuid == Self.debug }
        guard error == nil, chars.contains(where: { $0.uuid == Self.command }) else {
            fail(peripheral)
            return
        }
        for ch in chars { peripheral.setNotifyValue(true, for: ch) }
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateNotificationStateFor characteristic: CBCharacteristic,
                    error: Error?) {
        guard characteristic.uuid == Self.command else { return }
        if error == nil, characteristic.isNotifying {
            watchdog?.cancel()
            state = .connected
        } else {
            fail(peripheral)
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard error == nil, let data = characteristic.value,
              let text = String(data: data, encoding: .utf8) else { return }
        switch characteristic.uuid {
        case Self.command: onCommand?(text)
        case Self.debug: onDebug?(text)
        default: break
        }
    }
}
