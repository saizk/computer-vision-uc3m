def create_annotations(masks, is_crowd=0):

    annotations = [get_annotation(mask) for mask in masks]
    annotations = [ann for ann in annotations if ann]

    return [{
        "id": idx + 1,
        "image_id": idx + 1,  # int
        "segmentation": segmentation,  # list
        "area": area,  # int
        "bbox": bbox,  # list
        "category_id": category_id,
        "is_crowd": is_crowd
    } for idx, (segmentation, bbox, area, category_id) in enumerate(annotations)]