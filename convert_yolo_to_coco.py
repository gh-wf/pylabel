from pylabel import importer

path_to_annotations = "H:/python/pylabel/yololabels/"

#Identify the path to get from the annotations to the images 
#path_to_images = "H:/python/pylabel/yololabels/"

img_height, img_width = 3888, 5184

#Import the dataset into the pylable schema 
#Class names are defined here https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml
yoloclasses = [ 'crossarm','cutouts', 'insulator','pole','transformers','background_structure']


dataset = importer.ImportYoloV5Labels(path=path_to_annotations, img_height=img_height, img_width=img_width, cat_names=yoloclasses,
    name="epri_drone_dist")

print(dataset.df.head(5))

print(f"Number of images: {dataset.analyze.num_images}")
print(f"Number of classes: {dataset.analyze.num_classes}")
print(f"Classes:{dataset.analyze.classes}")
print(f"Class counts:\n{dataset.analyze.class_counts}")


dataset.export.ExportToCoco(cat_id_index=1)