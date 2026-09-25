// Step 1: Hello, GPU. Prints what this Mac's chip and GPU can do.

import Foundation

do {
    print(render(try gatherReport()))
} catch {
    FileHandle.standardError.write("hello_gpu: \(error)\n".data(using: .utf8)!)
    exit(1)
}
