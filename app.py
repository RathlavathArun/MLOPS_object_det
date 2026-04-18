import gradio as gr
import requests

API = "http://127.0.0.1:8000"


def detect(video, target):
    with open(video, "rb") as f:
        res = requests.post(
            f"{API}/upload_video",
            files={"file": f},
            data={"target": target}
        )

    return "output.mp4"


def live_stream(target):
    url = f"{API}/detect_video"
    if target:
        url += f"?target={target}"

    return f'<img src="{url}" width="100%">'


with gr.Blocks() as demo:

    gr.Markdown("# 🔥 Object Detection Dashboard")

    mode = gr.Radio(
        ["Upload", "Live"],
        value="Upload",
        label="Select Mode"
    )

    # -------- UPLOAD --------
    with gr.Row(visible=True) as upload_row:

        with gr.Column():
            video = gr.Video(height=350)
            target = gr.Textbox(label="Object (person/car)")
            run = gr.Button("Run Detection")

        with gr.Column():
            output = gr.Video(height=350)

    # -------- LIVE --------
    with gr.Row(visible=False) as live_row:

        with gr.Column():
            live_target = gr.Textbox(label="Object")
            start = gr.Button("Start")
            stop = gr.Button("Stop")

        with gr.Column():
            live_output = gr.HTML()

    def switch(m):
        return (
            gr.update(visible=m == "Upload"),
            gr.update(visible=m == "Live")
        )

    mode.change(switch, inputs=mode, outputs=[upload_row, live_row])

    run.click(detect, inputs=[video, target], outputs=output)
    start.click(live_stream, inputs=live_target, outputs=live_output)
    stop.click(lambda: "", outputs=live_output)

if __name__ == "__main__":
    demo.launch()