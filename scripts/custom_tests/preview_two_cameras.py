#!/usr/bin/env python

import sys

import cv2


ROTATE_OPTIONS = {
    "rotate0": 0,
    "rotate90": 90,
    "rotate180": 180,
    "rotate270": 270,
    "rot0": 0,
    "rot90": 90,
    "rot180": 180,
    "rot270": 270,
}


def find_available_cameras(max_index: int = 10):
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
        cap.release()
    return available


def apply_transform(frame, rotation_deg: int = 0, mirror_axis: str | None = None):
    if rotation_deg % 360 == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_deg % 360 == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_deg % 360 == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if mirror_axis == "x":
        frame = cv2.flip(frame, 0)
    elif mirror_axis == "y":
        frame = cv2.flip(frame, 1)

    return frame


def parse_camera_input(raw: str, available_ids):
    tokens = raw.split()
    if not tokens:
        raise ValueError("No camera input provided.")

    ordered = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        try:
            cam_id = int(token)
        except ValueError:
            raise ValueError(f"Expected a camera id, got '{token}'.")

        if cam_id not in available_ids:
            raise ValueError(f"Camera id {cam_id} is not available. Choose from: {available_ids}")

        if any(entry["id"] == cam_id for entry in ordered):
            raise ValueError(f"Duplicate camera id {cam_id} is not allowed.")

        entry = {"id": cam_id, "rotation": 0, "mirror": None}
        i += 1

        while i < len(tokens):
            next_token = tokens[i].lower()

            if next_token in ROTATE_OPTIONS:
                entry["rotation"] = ROTATE_OPTIONS[next_token]
                i += 1
                continue

            if next_token in {"xmirror", "ymirror"}:
                entry["mirror"] = next_token.replace("mirror", "")
                i += 1
                continue

            break

        ordered.append(entry)

    if not 1 <= len(ordered) <= 3:
        raise ValueError("Please enter between 1 and 3 camera ids.")

    return ordered


def ask_for_camera_ids(available_ids):
    print(f"Available camera ids: {available_ids}")
    print("Enter 1 to 3 camera ids, with optional transform flags after each id.")
    print("Examples:")
    print("  1 2 0")
    print("  1 rotate180 xmirror 2 0 ymirror")
    print("  2 rotate90 ymirror 0")
    print("Accepted rotations: rotate0, rotate90, rotate180, rotate270")
    print("Accepted mirrors: xmirror, ymirror")

    while True:
        raw = input("Which camera(s) to preview? ").strip()
        if not raw:
            print("No camera id entered. Please try again.")
            continue

        try:
            ordered = parse_camera_input(raw, available_ids)
            return ordered
        except ValueError as exc:
            print(f"{exc}")


def resize_frame_to_height(frame, target_h: int):
    h, w = frame.shape[:2]
    if h == 0:
        return frame
    new_w = max(1, int(w * target_h / h))
    return cv2.resize(frame, (new_w, target_h))


def main() -> None:
    available_ids = find_available_cameras()
    if not available_ids:
        print("No camera found on this machine.")
        sys.exit(1)

    selected = ask_for_camera_ids(available_ids)

    caps = []
    for entry in selected:
        cam_id = entry["id"]
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"Couldn't open camera {cam_id}.")
            for existing_cap in caps:
                existing_cap.release()
            sys.exit(1)
        caps.append((cam_id, cap, entry["rotation"], entry["mirror"]))

    print(f"Previewing camera(s) in order: {[entry['id'] for entry in selected]} — press q to close")

    try:
        while True:
            frames = []
            for cam_id, cap, rotation_deg, mirror_axis in caps:
                ok, frame = cap.read()
                if not ok:
                    print(f"Failed to read from camera {cam_id}.")
                    return

                frame = apply_transform(frame, rotation_deg, mirror_axis)
                frame = resize_frame_to_height(frame, 480)
                label = f"Camera {cam_id}"
                if rotation_deg:
                    label += f" rot{rotation_deg}"
                if mirror_axis:
                    label += f" {mirror_axis}mirror"

                cv2.putText(
                    frame,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                frames.append(frame)

            combined = cv2.hconcat(frames)
            cv2.imshow("Camera Preview", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for _, cap, _, _ in caps:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
