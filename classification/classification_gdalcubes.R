library(gdalcubes)
library(sf)
library(parallel)
library(rstac)
library(randomForest)
library(caret)
library(xgboost)
library(readr)
library(terra)
library(jsonlite)

# detect the number of CPU cores on the current host
CORES <- parallel::detectCores()
gdalcubes_options(parallel = CORES)

# Training data in GeoJSON format
training_sites <- read_sf("./train_data/train_dat.geojson")
assets <- c("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "SCL")

bbox <- st_bbox(training_sites)
# Define the bounding box for the training area
aot_test <- st_bbox(c(xmin = 388831.6, ymin = 5698900.1, xmax = 398063.3, ymax = 5718133.7), crs = st_crs(25832))

# Define the bounding box in EPSG:4326 The coordinates form a bounding box in which Paderborn is located
aoi_target <- st_bbox(c(xmin = 8.6638, ymin = 51.6612, xmax = 8.8410, ymax = 51.7815), crs = st_crs(4326))

# Conversion to an sf object
bbox_target_sf <- st_as_sfc(aoi_target)

# Transform the coordinates to EPSG:25832. 
bbox_target_transformed <- st_transform(bbox_target_sf, crs = st_crs(25832))

# Extract the transformed coordinates
bbox_target_utm <- st_bbox(bbox_target_transformed)

# Output of the transformed coordinates
print(bbox_target_utm)

# The transfromation is not really necessary here, it is also carried out universally in the train_model function.

#################Get picture#################
# STAC-Server URL
s <- stac("https://earth-search.aws.element84.com/v0")

# Perform STAC search
items_training <- s |>
  stac_search(collections = "sentinel-s2-l2a-cogs",
              bbox = c(bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]), 
              datetime = "2021-06-01/2021-06-15") |>
  post_request() |> items_fetch(progress = FALSE)

# Number of elements found
length(items_training$features)
s2_collection_training <- stac_image_collection(items_training$features, asset_names = assets, property_filter = function(x) {x[["eo:cloud_cover"]] < 20})
s2_collection_training
print(s2_collection_training)

# Define the bounding box based on the actual values of the data cube
aot_cube <- cube_view(extent = s2_collection_training, srs = "EPSG:25832", dx = 10, dy = 10, dt = "P1M",
                      aggregation = "median", resampling = "average")
aot_cube

# Create the data cube with the defined view
data_cube_training <- raster_cube(s2_collection_training, aot_cube) %>%
  select_bands(c("B02", "B03", "B04", "B08")) %>%
  gdalcubes::crop(extent = list(left = aot_test["xmin"],
                                right = aot_test["xmax"],
                                bottom = aot_test["ymin"],
                                top = aot_test["ymax"],
                                t0 = "2021-06", t1 = "2021-06"),
                  snap = "near")
data_cube_training

######################AOI############
# AOI also Area of Intereast is our area of interest, on which we want to train our model later. 
# Data acquisition is standard, a time step with the bands B02/B03/B04/B08

aoi_target
bbox_target_utm
# STAC-Server URL
s <- stac("https://earth-search.aws.element84.com/v0")

# Perform STAC search
items_training_aoi <- s |>
  stac_search(collections = "sentinel-s2-l2a-cogs",
              bbox = c(aoi_target["xmin"], aoi_target["ymin"], aoi_target["xmax"], aoi_target["ymax"]), 
              datetime = "2021-06-01/2021-06-15") |>
  post_request() |> items_fetch(progress = FALSE)

# Number of elements found
length(items_training_aoi$features)
s2_collection_aoi <- stac_image_collection(items_training_aoi$features, asset_names = assets, property_filter = function(x) {x[["eo:cloud_cover"]] < 20})
s2_collection_aoi

cube_aoi <- cube_view(extent = s2_collection_aoi, srs = "EPSG:25832", dx = 10, dy = 10, dt = "P1M",
                      aggregation = "median", resampling = "average",)

cube_aoi

aoi_training <- raster_cube(s2_collection_aoi, cube_aoi) %>%
  select_bands(c("B02", "B03", "B04", "B08")) %>%
  gdalcubes::crop(extent = list(left = bbox_target_utm["xmin"],
                                right = bbox_target_utm["xmax"],
                                bottom = bbox_target_utm["ymin"],
                                top = bbox_target_utm["ymax"],
                                t0 = "2021-06", t1 = "2021-06"),
                  snap = "near")



# Function for training a model based on a data cube and training data
train_model <- function(aot_cube, train_dat, method) {
  # Extracts the coordinate system (CRS) of the data cube
  cube_crs <- gdalcubes::srs(aot_cube)
  crs_data <- as.numeric(gsub("EPSG:", "", cube_crs))
  
  # Transforms the training data into the coordinate system of the data cube  
  train_dat <- sf::st_transform(train_dat, crs = crs_data)
  print(train_dat)
  
  message("Combining training data with cube data . . . .")
  
  # Extracts the data from the data cube based on the geometries of the training data
  extraction <- extract_geom(
    cube = aot_cube,
    sf = train_dat
  )
  message("Trainingsdata extracted ....")
  
  if (nrow(extraction) == 0) {
    stop("No data extracted. Check if the bounding boxes of the training data and the data cube overlap.")
  }
  print(extraction)

  # Merges the extracted data with the training data based on the IDs  
  message("Now merging trainingsdata with aoi data ....")
  train_dat$PolyID <- seq_len(nrow(train_dat))
  print(train_dat)
  extraction <- merge(extraction, train_dat, by.x = "FID", by.y = "PolyID")
  print(extraction)
  message("Extraction merged with trainingsdata ....")
  
  # Prepares the training data for model training
  message("Now preparing the trainingdata for the modeltraining ....")
  predictors <- c("B02", "B03", "B04", "B08") 
  # Teilt die Daten in Trainings- und Testdatensätze
  train_ids <- createDataPartition(extraction$FID, p = 0.2, list = FALSE)
  train_data <- extraction[train_ids, ]
  train_data <- train_data[complete.cases(train_data[, predictors]), ]
  train_data <- as.data.frame(train_data)
  message("Trainingdata prepared . . . .")
  
  # Ensures that the column names are correct and separates the data into predictors (x) and target variable (y)  
  x <- train_data[, predictors]
  y <- train_data$Label
  
  print(head(x))
  print(head(y))
  
  # Trains the model with the prepared training data. Either the Random Forest, SVM or the XGBoost is used.   
  message("Now training the model . . . .")
  if(method == "xgbTree"){
    params <- list(
      booster = "gbtree",
      objective = "multi:softmax",
      num_class = length(unique(y)), # Anzahl der Klassen
      eta = 0.3,
      max_depth = 6
    )
    model <- caret::train(
      x, y,
      method = method,
      trControl = trainControl(method = "cv", number = 5),
      tuneGrid = expand.grid(
        nrounds = 100,
        max_depth = params$max_depth,
        eta = params$eta,
        gamma = 0,
        colsample_bytree = 1,
        min_child_weight = 1,
        subsample = 1
      )
    )
  }else{
    model <- caret::train(x, y,
                          method = method,
                          importance = TRUE,
                          ntree = 50)
  }

  #Accuracy of the model
  message("Model trained, accuracy: ", model$results$Accuracy)
  return(model)
}

# Train the model with the data_cube and the training data
model <- train_model(data_cube_training, training_sites, method = "rf")
model_svm <- train_model(data_cube_training, training_sites, method = "svmRadial")
model_xgboost <- train_model(data_cube_training, training_sites, method = "xgbTree")


# Prediction function
prediction <- function(model, aoi_cube) {
  tryCatch({
    prediction <- predict(aoi_cube, model)
    print(prediction)
    message("Prediction calculated ....")
    message(gdalcubes::as_json(prediction))
    
    return(prediction)
  },
  error = function(e){
    message("Error in classification")
    message(conditionMessage(e))
  })
}

# Performs the prediction with the model and the aoi-cube
prediction_rf <- prediction(model = model, aoi_cube = aoi_training)
prediction_svm <- prediction(model = model_svm, aoi_cube = aoi_training)
prediction_xgboost <- prediction(model = model_xgboost, aoi_cube = aoi_training)

prediction_raster <- terra::rast(gdalcubes::write_tif(prediction_rf))
prediction_raster_svm <- terra::rast(gdalcubes::write_tif(prediction_svm))
prediction_raster_xgboost <- terra::rast(gdalcubes::write_tif(prediction_xgboost))

plot(prediction_raster)
plot(prediction_raster_svm)
plot(prediction_raster_xgboost)
