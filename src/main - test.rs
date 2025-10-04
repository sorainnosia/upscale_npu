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
            window_size: 8,
            description: "Real-world photos (4x) - Best overall quality",
        },
        ModelInfo {
            name: "swin2SR-classical-sr-x4-64",
            url: "https://huggingface.co/Xenova/swin2SR-classical-sr-x4-64/resolve/main/onnx/model.onnx",
            scale: 4,
            window_size: 8,
            description: "Clean images (4x) - High quality",
        },
        ModelInfo {
            name: "swin2SR-lightweight-x2-64",
            url: "https://huggingface.co/Xenova/swin2SR-lightweight-x2-64/resolve/main/onnx/model.onnx",
            scale: 2,
            window_size: 8,
            description: "Lightweight (2x) - Fastest",
        },
        ModelInfo {
            name: "swin2SR-compressed-sr-x4-48",
            url: "https://huggingface.co/Xenova/swin2SR-compressed-sr-x4-48/resolve/main/onnx/model.onnx",
            scale: 4,
            window_size: 8,
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
    window_size: u32,
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
         .with_execution_providers([
             ort::execution_providers::DirectMLExecutionProvider::default().build()
         ])?
        .commit_from_file(&model_path)?;

    let (input_width, input_height) = img.dimensions();
    let original_width = input_width;
    let original_height = input_height;
    
    // Check if image is too large and resize if needed
    const MAX_DIMENSION: u32 = 512;
    let img = if input_width > MAX_DIMENSION || input_height > MAX_DIMENSION {
        println!("  ⚠️  Image too large! Resizing to fit within {}x{}", MAX_DIMENSION, MAX_DIMENSION);
        let scale = (MAX_DIMENSION as f32 / input_width.max(input_height) as f32).min(1.0);
        let new_width = (input_width as f32 * scale) as u32;
        let new_height = (input_height as f32 * scale) as u32;
        println!("  Resized to: {}x{}", new_width, new_height);
        img.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3)
    } else {
        img.clone()
    };

    // Pad image to be divisible by window_size
    let (padded_img, (input_width, input_height), (pad_right, pad_bottom)) = 
        pad_to_multiple(&img, model.window_size)?;
    
    if pad_right > 0 || pad_bottom > 0 {
        println!("  📐 Padded {}x{} -> {}x{} (divisible by {})", 
                 img.dimensions().0, img.dimensions().1,
                 input_width, input_height, model.window_size);
    }

    // Preprocess
    println!("  🔄 Processing...");
    let input_tensor = preprocess_image(&padded_img)?;
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

    let mut upscaled_img = postprocess_tensor(output_array)?;
    
    // Crop to remove padding from output (scaled by the model's scale factor)
    if pad_right > 0 || pad_bottom > 0 {
        let target_width = img.dimensions().0 * model.scale;
        let target_height = img.dimensions().1 * model.scale;
        upscaled_img = upscaled_img.crop_imm(0, 0, target_width, target_height);
        println!("  ✂️  Cropped output to: {}x{}", target_width, target_height);
    }
    
    let (output_width, output_height) = upscaled_img.dimensions();

    // Save
    let output_path = format!("output_{}.png", model.name);
    upscaled_img.save(&output_path)?;

    let duration = start.elapsed().as_secs_f32();

    Ok(ProcessStats {
        input_width: original_width,
        input_height: original_height,
        output_width,
        output_height,
        duration,
        output_path,
    })
}

fn pad_to_multiple(img: &DynamicImage, multiple: u32) -> Result<(DynamicImage, (u32, u32), (u32, u32))> {
    let (width, height) = img.dimensions();
    
    // Calculate padded dimensions
    let padded_width = ((width + multiple - 1) / multiple) * multiple;
    let padded_height = ((height + multiple - 1) / multiple) * multiple;
    
    let pad_right = padded_width - width;
    let pad_bottom = padded_height - height;
    
    // If no padding needed, return original
    if pad_right == 0 && pad_bottom == 0 {
        return Ok((img.clone(), (width, height), (0, 0)));
    }
    
    // Create padded image with reflection padding
    let mut padded = ImageBuffer::new(padded_width, padded_height);
    let rgb_img = img.to_rgb8();
    
    for y in 0..padded_height {
        for x in 0..padded_width {
            // Use reflection padding
            let src_x = if x < width {
                x
            } else {
                width - 1 - (x - width).min(width - 1)
            };
            
            let src_y = if y < height {
                y
            } else {
                height - 1 - (y - height).min(height - 1)
            };
            
            let pixel = rgb_img.get_pixel(src_x, src_y);
            padded.put_pixel(x, y, *pixel);
        }
    }
    
    Ok((DynamicImage::ImageRgb8(padded), (padded_width, padded_height), (pad_right, pad_bottom)))
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