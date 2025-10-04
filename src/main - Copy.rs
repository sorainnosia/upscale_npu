// Cargo.toml dependencies:
/*
[dependencies]
ort = { version = "2.0.0-rc.4", features = ["load-dynamic"] }
ndarray = "0.16"
image = "0.25"
anyhow = "1.0"
reqwest = { version = "0.12", features = ["blocking"] }
*/

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::Array4;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    ort::init().commit()?;

    // All available Swin2SR models
    let models = vec![
        ModelInfo {
            name: "swin2SR-realworld-sr-x4-64-bsrgan-psnr",
            url: "https://huggingface.co/Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr/resolve/main/onnx/model.onnx",
            scale: 4,
            description: "Real-world photos (4x) - Best overall quality",
        },
        ModelInfo {
            name: "swin2SR-classical-sr-x4-64",
            url: "https://huggingface.co/Xenova/swin2SR-classical-sr-x4-64/resolve/main/onnx/model.onnx",
            scale: 4,
            description: "Clean images (4x) - High quality",
        },
        ModelInfo {
            name: "swin2SR-lightweight-x2-64",
            url: "https://huggingface.co/Xenova/swin2SR-lightweight-x2-64/resolve/main/onnx/model.onnx",
            scale: 2,
            description: "Lightweight (2x) - Fastest",
        },
        ModelInfo {
            name: "swin2SR-compressed-sr-x4-48",
            url: "https://huggingface.co/Xenova/swin2SR-compressed-sr-x4-48/resolve/main/onnx/model.onnx",
            scale: 4,
            description: "Compressed/JPEG images (4x)",
        },
    ];

    println!("🎨 Swin2SR Multi-Model Comparison Tool\n");
    println!("Available models:");
    for (i, model) in models.iter().enumerate() {
        println!("  {}. {} - {}", i + 1, model.name, model.description);
    }
    println!("\nEnter model number to use (1-{}), or 'all' to compare all: ", models.len());

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim();

    let input_path = "input.jpg";
    let img = image::open(input_path)
        .context("Failed to load input.jpg. Please add an image to upscale!")?;

    if input.to_lowercase() == "all" {
        println!("\n🔄 Running comparison on all models...\n");
        for model in &models {
            match process_with_model(model, &img) {
                Ok(stats) => {
                    println!("✓ {}", model.name);
                    println!("  Time: {:.2}s | Output: {}x{}\n", 
                             stats.duration, stats.output_width, stats.output_height);
                }
                Err(e) => {
                    println!("✗ {} - Error: {}\n", model.name, e);
                }
            }
        }
    } else {
        let idx: usize = input.parse::<usize>()
            .context("Invalid number")?
            .saturating_sub(1);
        
        if idx >= models.len() {
            anyhow::bail!("Invalid model number");
        }

        let model = &models[idx];
        println!("\n🚀 Using: {}\n", model.name);
        
        let stats = process_with_model(model, &img)?;
        println!("\n✅ Complete!");
        println!("Time taken: {:.2}s", stats.duration);
        println!("Output: {}x{} -> {}x{} ({}x scale)", 
                 stats.input_width, stats.input_height,
                 stats.output_width, stats.output_height,
                 stats.output_width / stats.input_width);
        println!("Saved to: {}", stats.output_path);
    }

    Ok(())
}

struct ModelInfo {
    name: &'static str,
    url: &'static str,
    scale: u32,
    description: &'static str,
}

struct ProcessStats {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    duration: f32,
    output_path: String,
}

fn process_with_model(model: &ModelInfo, img: &DynamicImage) -> Result<ProcessStats> {
    let start = Instant::now();
    
    // Download model if needed
    let model_path = format!("{}.onnx", model.name);
    if !Path::new(&model_path).exists() {
        println!("  📥 Downloading {}...", model.name);
        download_model(model.url, &model_path)?;
    }

    // Load session
    println!("  ⚙️  Loading model...");
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // Uncomment for AMD Ryzen NPU:
        // .with_execution_providers([
        //     ort::execution_providers::DirectMLExecutionProvider::default().build()
        // ])?
        .commit_from_file(&model_path)?;

    let (input_width, input_height) = img.dimensions();
    
    // Preprocess
    println!("  🔄 Processing...");
    let input_tensor = preprocess_image(img)?;
    let input_value = Value::from_array(input_tensor)?;
	let input_name = &session.inputs[0].name.to_string();
    let output_name = &session.outputs[0].name.to_string();

    // Inference
    let outputs = session.run(ort::inputs![input_name.as_str() => input_value])?;

    // Postprocess
    let (output_shape, output_data) = outputs[output_name.as_str()]
        .try_extract_tensor::<f32>()?;
    
    let shape_vec = output_shape.as_ref().to_vec();
    let output_array = Array4::from_shape_vec(
        (shape_vec[0] as usize, shape_vec[1] as usize, 
         shape_vec[2] as usize, shape_vec[3] as usize),
        output_data.to_vec()
    )?;

    let upscaled_img = postprocess_tensor(output_array)?;
    let (output_width, output_height) = upscaled_img.dimensions();

    // Save
    let output_path = format!("output_{}.png", model.name);
    upscaled_img.save(&output_path)?;

    let duration = start.elapsed().as_secs_f32();

    Ok(ProcessStats {
        input_width,
        input_height,
        output_width,
        output_height,
        duration,
        output_path,
    })
}

fn download_model(url: &str, path: &str) -> Result<()> {
    let response = reqwest::blocking::get(url)?;
    let bytes = response.bytes()?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn preprocess_image(img: &DynamicImage) -> Result<Array4<f32>> {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    
    let mut tensor = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
    
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_img.get_pixel(x, y);
            tensor[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            tensor[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            tensor[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }
    }
    
    Ok(tensor)
}

fn postprocess_tensor(tensor: Array4<f32>) -> Result<DynamicImage> {
    let shape = tensor.shape();
    let (_, _, height, width) = (shape[0], shape[1], shape[2], shape[3]);
    
    let mut img_buffer = ImageBuffer::new(width as u32, height as u32);
    
    for y in 0..height {
        for x in 0..width {
            let r = (tensor[[0, 0, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (tensor[[0, 1, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (tensor[[0, 2, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            
            img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    
    Ok(DynamicImage::ImageRgb8(img_buffer))
}