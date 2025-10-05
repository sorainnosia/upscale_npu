# Upscale_NPU
A single or batch images file(s) upscaler using ONNX models

# Sample
<img width="1620" height="1000" alt="image" src="https://github.com/user-attachments/assets/debacd02-31f5-44e1-bc22-6de8d9e4426b" />

# Build on Windows
1. Download the required dll from `https://github.com/microsoft/onnxruntime/releases` and extract `onnxruntime.dll` and put in the root folder of the project
2. Run below command replace `<Path to DLL>` to the directory of the dll
```
    set ORT_DYLIB_PATH=<Path to DLL>\onnxruntime.dll
    cargo build --release
```
3. The program has 3 files `DirectML.dll`, `upscale_npu.exe`, `onnxruntime.dll`, upon run of upscale_npu.exe it will download the models from `https://huggingface.co/Xenova` repositories and use the model to upscale the input file

# Example of run
1. Command to upscale a single file
`upscale_npu.exe input.jpg output.jpg`
2. Command to upscale batch files in a folder
`upscale_npu.exe path_to_folder\folder path_to_folder\output`
