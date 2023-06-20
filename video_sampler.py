import cv2


def create_snippets_video(video_path, snippet_length, overlap_length):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    snippet_frames = snippet_length * fps
    overlap_frames = overlap_length * fps
    snippet_count = (total_frames - snippet_frames) // overlap_frames + 1

    buffer_frames = []
    snippet_num = 1
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        buffer_frames.append(frame)

        if len(buffer_frames) > snippet_frames:
            buffer_frames.pop(0)

        if len(buffer_frames) == snippet_frames:
            if frame_num % overlap_frames == 0 or frame_num == total_frames:
                snippet = buffer_frames.copy()
                write_snippet(snippet, snippet_num)
                snippet_num += 1

    cap.release()


def create_snippets_camera(snippet_length, overlap_length):
    # Open the webcam stream (0 represents the default webcam)
    cap = cv2.VideoCapture(0)

    fps = 30  # Assuming the webcam captures at 30 frames per second
    snippet_frames = snippet_length * fps
    overlap_frames = overlap_length * fps

    buffer_frames = []
    snippet_num = 1
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        buffer_frames.append(frame)

        if len(buffer_frames) > snippet_frames:
            buffer_frames.pop(0)

        if len(buffer_frames) == snippet_frames:
            if frame_num % overlap_frames == 0:
                snippet = buffer_frames.copy()
                write_snippet(snippet, snippet_num)
                snippet_num += 1

    cap.release()


def write_snippet(snippet_frames, snippet_num):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = snippet_frames[0].shape
    output_name = f'snippet_{snippet_num}.mp4'
    out = cv2.VideoWriter(output_name, fourcc, 30.0, (width, height))

    for frame in snippet_frames:
        out.write(frame)

    out.release()
    print(f'Snippet {output_name} created.')


# Usage
video_path = '/home/dev/workspace/sample_videos/30da536e-4848-4d54-8f2b-6fe1ad54be11.mp4'
snippet_length = 2  # in seconds
overlap_length = 1  # in seconds

# create_snippets_video(video_path, snippet_length, overlap_length)
create_snippets_camera(snippet_length, overlap_length)
