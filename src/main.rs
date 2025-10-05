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

    // All available super-resolution models
    let models = vec![
        // === Swin2SR Models (Latest) ===
        ModelInfo {
            name: "swin2SR-realworld-sr-x4-64-bsrgan-psnr",
            url: "https://huggingface.co/Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr/resolve/main/onnx/model.onnx",
            scale: 4,
            window_size: 8,
            description: "Real-world photos (4x) - Best overall quality",
            category: "Swin2SR",
        },
        ModelInfo {
            name: "swin2SR-classical-sr-x4-64",
            url: "https://huggingface.co/Xenova/swin2SR-classical-sr-x4-64/resolve/main/onnx/model.onnx",
            scale: 4,
            window_size: 8,
            description: "Clean images (4x) - High quality",
            category: "Swin2SR",
        },
        ModelInfo {
            name: "swin2SR-lightweight-x2-64",
            url: "https://huggingface.co/Xenova/swin2SR-lightweight-x2-64/resolve/main/onnx/model.onnx",
            scale: 2,
            window_size: 8,
            description: "Lightweight (2x) - Fastest",
            category: "Swin2SR",
        },
        ModelInfo {
            name: "swin2SR-compressed-sr-x4-48",
            url: "https://huggingface.co/Xenova/swin2SR-compressed-sr-x4-48/resolve/main/onnx/model.onnx",
            scale: 4,
            window_size: 8,
            description: "Compressed/JPEG images (4x)",
            category: "Swin2SR",
        },
        
        // === APISR Models (High Quality GAN-based) ===
        ModelInfo {
            name: "2x_APISR_RRDB_GAN_generator",
            url: "https://huggingface.co/Xenova/2x_APISR_RRDB_GAN_generator-onnx/resolve/main/onnx/model.onnx",
            scale: 2,
            window_size: 1, // No window requirement
            description: "APISR GAN 2x - Excellent for anime/illustrations",
            category: "APISR",
        },
        ModelInfo {
            name: "4x_APISR_GRL_GAN_generator",
            url: "https://huggingface.co/Xenova/4x_APISR_GRL_GAN_generator-onnx/resolve/main/onnx/model.onnx",
            scale: 4,
            window_size: 1, // No window requirement
            description: "APISR GAN 4x - High quality for anime/illustrations",
            category: "APISR",
        },
        /*
        // === RealESRGAN Models ===
        ModelInfo {
            name: "Real-ESRGAN-x4plus",
            url: "https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx",
            scale: 4,
            window_size: 1, // No window requirement
            description: "RealESRGAN 4x - General purpose, great quality",
            category: "RealESRGAN",
        },
        ModelInfo {
            name: "Real-ESRGAN-General-x4v3",
            url: "https://huggingface.co/qualcomm/Real-ESRGAN-General-x4v3/resolve/main/Real-ESRGAN-General-x4v3.onnx",
            scale: 4,
            window_size: 1,
            description: "RealESRGAN v3 (4x) - Improved general quality",
            category: "RealESRGAN",
        },
        ModelInfo {
            name: "lightweight-real-ESRGAN-anime",
            url: "https://huggingface.co/xiongjie/lightweight-real-ESRGAN-anime/resolve/main/model.onnx",
            scale: 4,
            window_size: 1,
            description: "Lightweight anime upscaler (4x) - Very fast",
            category: "RealESRGAN",
        },
        */
        // === SwinIR Models (Earlier version) ===
        ModelInfo {
            name: "SwinIR-realworld-x4",
            url: "https://huggingface.co/rocca/swin-ir-onnx/resolve/main/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx",
            scale: 4,
            window_size: 8,
            description: "SwinIR real-world (4x) - Good for degraded images",
            category: "SwinIR",
        },
    ];

    println!("🎨 Enhanced Super-Resolution Image Upscaler\n");
    
    let args: Vec<String> = std::env::args().collect();
    
    // Model selection
    let selected_model_idx = if args.len() > 3 {
        args[3].parse::<usize>().unwrap_or(0).min(models.len() - 1)
    } else {
        // Display available models
        println!("📋 Available Models:\n");
        
        let mut current_category = "";
        for (idx, model) in models.iter().enumerate() {
            if model.category != current_category {
                println!("\n{}", model.category);
                println!("{}", "=".repeat(40));
                current_category = model.category;
            }
            println!("[{}] {} - {}", idx, model.name, model.description);
        }
        
        println!("\n{}", "=".repeat(60));
        println!("Select model number (default: 0): ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        input.trim().parse::<usize>().unwrap_or(0).min(models.len() - 1)
    };
    
    let model = &models[selected_model_idx];
    println!("\n✨ Using: {} ({}x upscale)\n", model.name, model.scale);
    
    // Get input path from args or prompt
    let input_path = if args.len() > 1 {
        args[1].clone()
    } else {
        println!("Enter input path (file or folder):");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        input.trim().to_string()
    };
    
    // Get output path from args or prompt
    let output_path = if args.len() > 2 {
        args[2].clone()
    } else {
        println!("Enter output path (file or folder):");
        let mut output = String::new();
        std::io::stdin().read_line(&mut output)?;
        output.trim().to_string()
    };
    
    let input_metadata = std::fs::metadata(&input_path)
        .context("Failed to read input path. Please check if it exists.")?;
    
    // Check if input is a file or directory
    if input_metadata.is_file() {
        // Single file mode
        println!("📄 Processing single file...\n");
        
        let img = image::open(&input_path)
            .context("Failed to load input image")?;
        
        let start = Instant::now();
        let stats = process_with_model_to_path(model, &img, &output_path)?;
        
        println!("\n✅ Complete!");
        println!("Time taken: {:.2}s", stats.duration);
        println!("Output: {}x{} -> {}x{} ({}x scale)", 
                 stats.input_width, stats.input_height,
                 stats.output_width, stats.output_height,
                 model.scale);
        println!("Saved to: {}", output_path);
    } else {
        // Batch folder mode
        println!("📁 Processing folder...\n");
        
        // Create output folder if it doesn't exist
        std::fs::create_dir_all(&output_path)
            .context("Failed to create output directory")?;
        
        // Find all image files
        let image_extensions = ["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"];
        let mut image_files = Vec::new();
        
        for entry in std::fs::read_dir(&input_path)
            .context("Failed to read input directory")? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if let Some(ext_str) = ext.to_str() {
                        if image_extensions.contains(&ext_str.to_lowercase().as_str()) {
                            image_files.push(path);
                        }
                    }
                }
            }
        }
        
        if image_files.is_empty() {
            anyhow::bail!("No image files found in input folder");
        }
        
        println!("Found {} image(s) to process\n", image_files.len());
        
        let total_start = Instant::now();
        let mut successful = 0;
        let mut failed = 0;
        
        for (idx, img_path) in image_files.iter().enumerate() {
            let filename = img_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            
            println!("[{}/{}] Processing: {}", idx + 1, image_files.len(), filename);
            
            match image::open(img_path) {
                Ok(img) => {
                    let output_filename = img_path.file_stem()
                        .and_then(|n| n.to_str())
                        .unwrap_or("output");
                    let output_file_path = format!("{}/{}_{}x.png", output_path, output_filename, model.scale);
                    
                    match process_with_model_to_path(model, &img, &output_file_path) {
                        Ok(stats) => {
                            println!("  ✓ Complete - {:.2}s | {}x{} -> {}x{}", 
                                     stats.duration,
                                     stats.input_width, stats.input_height,
                                     stats.output_width, stats.output_height);
                            successful += 1;
                        }
                        Err(e) => {
                            println!("  ✗ Error: {}", e);
                            failed += 1;
                        }
                    }
                }
                Err(e) => {
                    println!("  ✗ Failed to load: {}", e);
                    failed += 1;
                }
            }
            println!();
        }
        
        let total_duration = total_start.elapsed().as_secs_f32();
        println!("========================================");
        println!("Batch processing complete!");
        println!("Total time: {:.2}s", total_duration);
        println!("Successful: {} | Failed: {}", successful, failed);
        println!("Output folder: {}", output_path);
    }
    
    Ok(())
}

struct ModelInfo {
    name: &'static str,
    url: &'static str,
    scale: u32,
    window_size: u32,
    description: &'static str,
    category: &'static str,
}

struct ProcessStats {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    duration: f32,
    output_path: String,
}

fn process_with_model_to_path(model: &ModelInfo, img: &DynamicImage, output_file_path: &str) -> Result<ProcessStats> {
    let start = Instant::now();
    
    // Download model if needed
    let model_path = format!("./models/{}.onnx", model.name);
    if !Path::new(&model_path).exists() {
        println!("  📥 Downloading {}...", model.name);
        download_model(model.url, &model_path)?;
    }

    // Load session (cache this for better performance in batch processing)
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // Uncomment for AMD Ryzen NPU or DirectML support:
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

    // Pad image to be divisible by window_size (if model requires it)
    let (padded_img, (input_width, input_height), (pad_right, pad_bottom)) = 
        if model.window_size > 1 {
            pad_to_multiple(&img, model.window_size)?
        } else {
            (img.clone(), img.dimensions(), (0, 0))
        };

    // Preprocess
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
    }
    
    let (output_width, output_height) = upscaled_img.dimensions();

    // Save
    upscaled_img.save(output_file_path)?;

    let duration = start.elapsed().as_secs_f32();

    Ok(ProcessStats {
        input_width: original_width,
        input_height: original_height,
        output_width,
        output_height,
        duration,
        output_path: output_file_path.to_string(),
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
	std::fs::create_dir_all("./models");
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
