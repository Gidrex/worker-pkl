{
  description = "RTSP Camera Emulation Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Script to run everything in one go
        fake-cam-script = pkgs.writeShellScriptBin "fake-cam" ''
          # 1. Start RTSP Server in background
          echo "Starting RTSP Server (mediamtx)..."
          ${pkgs.mediamtx}/bin/mediamtx > /dev/null 2>&1 &
          SERVER_PID=$!

          # Cleanup function (Ctrl+C)
          cleanup() {
            echo "Stopping RTSP Server..."
            kill $SERVER_PID
          }
          trap cleanup EXIT

          # Wait for server to start
          sleep 1

          # 2. Start stream
          VIDEO_FILE="./videos/КПП1_p2.avi" 

          if [ ! -f "$VIDEO_FILE" ]; then
            echo "Error: Video file $VIDEO_FILE not found!"
            exit 1
          fi

          echo "Streaming $VIDEO_FILE to rtsp://localhost:8554/cam1"
          echo "Press Ctrl+C to stop."

          ${pkgs.ffmpeg}/bin/ffmpeg \
            -re \
            -stream_loop -1 \
            -i "$VIDEO_FILE" \
            -c copy \
            -f rtsp \
            rtsp://localhost:8554/cam1
        '';

      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [ pkgs.ffmpeg pkgs.mediamtx fake-cam-script ];

          shellHook = ''
            echo "--------------------------------------------------------"
            echo "RTSP Environment Ready"
            echo "Tools available: ffmpeg, mediamtx"
            echo ""
            echo "Usage options:"
            echo "1. Run automated stream:  fake-cam"
            echo "2. Manual setup:"
            echo "   - Term 1: mediamtx"
            echo "   - Term 2: ffmpeg -re ... -f rtsp rtsp://localhost:8554/cam1"
            echo "--------------------------------------------------------"
          '';
        };
      });
}
