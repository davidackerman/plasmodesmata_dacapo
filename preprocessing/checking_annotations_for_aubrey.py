# %%
import yaml
import json
import urllib.parse
import os

# Create output directory for JSON files
output_dir = "tmp_proofreading_jsons"
os.makedirs(output_dir, exist_ok=True)

best_networks = {"jrc_22ak351-leaf-2lb": ["finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-2l_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00005_bs_2__1", "iteration_75000"],"jrc_22ak351-leaf-3mb": ["finetuned_3d_lsdaffs_weight_ratio_0.5_combined_healthy_and_gall_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00005_bs_2__0", "iteration_425000"],"jrc_22ak351-leaf-3rb": ["finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3r_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00005_bs_2__0", "iteration_250000"]}
for dataset in ["jrc_22ak351-leaf-2lb", "jrc_22ak351-leaf-3mb","jrc_22ak351-leaf-3rb"]:
    run_name, iteration = best_networks[dataset]
    manual_annotations = f"precomputed://https://cellmap-vm1.int.janelia.org/nrs/ackermand/cellmap/plasmodesmata/neuroglancer_annotations/{dataset}/removed_annotations"
    raw_path = f"https://cellmap-vm1.int.janelia.org/nrs/data/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8"
    training_validation_test_roi_yaml = f"./annotations/{dataset}/training_validation_test_roi_info.yaml"
    with open(training_validation_test_roi_yaml, 'r') as file:
        roi_info = yaml.safe_load(file)

    # Create neuroglancer link with ROI annotation layers
    def parse_roi_coordinate(coord_str):
        """Parse coordinate string like '2242-4795' into start and end values"""
        start, end = map(int, coord_str.split('-'))
        return start, end

    def create_neuroglancer_link_with_rois(roi_info, raw_path, manual_annotations, dataset, run_name, iteration):
        """Create a neuroglancer link with individual annotation layers for each ROI"""
        
        # Add each ROI as a separate annotation layer
        validation_test_rois = roi_info.get('rois_to_split', {}).get('validation_test', [])
        
        layers = []
        
        # Add raw data as the first layer
        raw_layer = {
            "type": "image",
            "name": "raw_data",
            "source": {
                "url": f"zarr://{raw_path}"
            },
            "shader": "#uicontrol float brightness slider(min=-1, max=1, default=0)\n#uicontrol float contrast slider(min=-3, max=3, default=0)\nvoid main() {\n  emitGrayscale(toNormalized(getDataValue()) + brightness, 1.0 + contrast);\n}",
            "visible": True
        }
        layers.append(raw_layer)
        
        # Add manual annotations as the second layer
        manual_annotations_layer = {
            "type": "annotation",
            "name": "manual_annotations",
            "source": {
                "url": manual_annotations
            },
            "annotationColor": "#ffff00",  # yellow color for manual annotations
            "visible": True
        }
        layers.append(manual_annotations_layer)
        
        # Calculate average center position of all ROIs
        total_x_center = 0
        total_y_center = 0
        total_z_center = 0
        roi_count = 0
        
        for roi in validation_test_rois:
            roi_name = roi['roi_name']
            
            # Parse coordinates (these are in nm, need to convert to voxel coordinates)
            x_start, x_end = parse_roi_coordinate(roi['x'])
            y_start, y_end = parse_roi_coordinate(roi['y'])
            z_start, z_end = parse_roi_coordinate(roi['z'])
            
            # Convert from nm to voxel coordinates (resolution is 8nm)
            resolution = roi_info.get('resolution', 8)
            x_start_voxel = x_start // resolution
            x_end_voxel = x_end // resolution
            y_start_voxel = y_start // resolution
            y_end_voxel = y_end // resolution
            z_start_voxel = z_start // resolution
            z_end_voxel = z_end // resolution
            
            # Calculate center of this ROI and add to totals
            roi_center_x = (x_start_voxel + x_end_voxel) / 2
            roi_center_y = (y_start_voxel + y_end_voxel) / 2
            roi_center_z = (z_start_voxel + z_end_voxel) / 2
            
            total_x_center += roi_center_x
            total_y_center += roi_center_y
            total_z_center += roi_center_z
            roi_count += 1
            
            # Determine color based on ROI type
            roi_type = roi.get('type_if_do_not_split', 'unknown')
            color = "#ff0000" if roi_type == 'validation' else "#00ff00"  # red for validation, green for test
            
            # Create annotation for this ROI (bounding box)
            # Use the proper neuroglancer format for bounding box annotations
            annotation_data = [
                {
                    "pointA": [x_start_voxel, y_start_voxel, z_start_voxel],
                    "pointB": [x_end_voxel, y_end_voxel, z_end_voxel], 
                    "type": "axis_aligned_bounding_box",
                    "id": roi_name,
                    "description": f"ROI {roi_name} ({roi_type}) - nm: [{x_start}-{x_end}, {y_start}-{y_end}, {z_start}-{z_end}]"
                }
            ]
            
            # Create layer for this ROI
            roi_layer = {
                "type": "annotation",
                "name": f"ROI_{roi_name}_{roi_type}",
                "source": "local://annotations",
                "tab": "annotations",
                "annotationColor": color,
                "annotations": annotation_data,
                "visible": True
            }
            
            layers.append(roi_layer)
            
            # Add prediction layer for this ROI
            validation_or_test = roi_type  # This should be 'validation' or 'test'
            prediction_url = f"https://cellmap-vm1.int.janelia.org/nrs/ackermand/predictions/leaf-gall/{dataset}/{dataset}.zarr/processed/{dataset}/{validation_or_test}/{run_name}/{roi_name}/{iteration}/"
            
            prediction_layer = {
                "type": "segmentation",
                "name": f"prediction_{roi_name}_{roi_type}",
                "source": {
                    "url": f"zarr://{prediction_url}"
                },
                "visible": True,  # Turn predictions on
                "opacity": 0.7
            }
            
            layers.append(prediction_layer)
        
        # Calculate average center position
        if roi_count > 0:
            avg_center_x = total_x_center / roi_count
            avg_center_y = total_y_center / roi_count
            avg_center_z = total_z_center / roi_count
            center_position = [avg_center_x, avg_center_y, avg_center_z]
        else:
            center_position = [7000, 7000, 10000]  # fallback position
        
        # Base neuroglancer state
        ng_state = {
            "dimensions": {
                "x": [8e-9, "m"],
                "y": [8e-9, "m"], 
                "z": [8e-9, "m"]
            },
            "position": center_position,
            "crossSectionScale": 1,
            "projectionOrientation": [0, 0, 0, 1],
            "projectionScale": 10000,
            "layers": layers,
            "selectedLayer": {
                "visible": True,
                "layer": layers[0]["name"] if layers else None
            },
            "layout": "4panel"
        }
        
        # Write state to JSON file
        json_filename = f"{dataset}_neuroglancer_state.json"
        json_filepath = os.path.join("tmp_proofreading_jsons", json_filename)
        
        with open(json_filepath, 'w') as f:
            json.dump(ng_state, f, indent=2)
        
        # Create the neuroglancer URL pointing to the JSON file
        json_url = f"https://cellmap-vm1.int.janelia.org/nrs/ackermand/scicompsoft_home/Programming/plasmodesmata_dacapo/preprocessing/{output_dir}/{json_filename}"
        neuroglancer_url = f"https://neuroglancer-demo.appspot.com/#!{json_url}"
        
        return neuroglancer_url, ng_state, roi_count, json_filepath

    # Generate the neuroglancer link
    neuroglancer_url, ng_state, total_rois, json_file = create_neuroglancer_link_with_rois(roi_info, raw_path, manual_annotations, dataset, run_name, iteration)

    # Print results for each dataset
    print(f"\n{dataset}:")
    print(f"  JSON file: {json_file}")
    print(f"  Neuroglancer URL: {neuroglancer_url}")
    print(f"  Total ROIs: {total_rois}")

# %%
