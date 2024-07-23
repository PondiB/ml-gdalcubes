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
library(tensorflow)
library(keras)


# detect the number of CPU cores on the current host
CORES <- parallel::detectCores()
gdalcubes_options(parallel = CORES)

# The data collection and processing is exactly the same as in the first example. 
# Please look there for more information. 

training_sites <- read_sf("./train_data/train_dat.geojson")

assets <- c("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "SCL")

bbox <- st_bbox(training_sites)

aot_test <- st_bbox(c(xmin = 388831.6, ymin = 5698900.1, xmax = 398063.3, ymax = 5718133.7), crs = st_crs(25832))

aoi_target <- st_bbox(c(xmin = 8.6638, ymin = 51.6612, xmax = 8.8410, ymax = 51.7815), crs = st_crs(4326))

bbox_target_sf <- st_as_sfc(aoi_target)

bbox_target_transformed <- st_transform(bbox_target_sf, crs = st_crs(25832))

bbox_target_utm <- st_bbox(bbox_target_transformed)

print(bbox_target_utm)


#################Get picture#################
# STAC-Server URL
s <- stac("https://earth-search.aws.element84.com/v0")

# STAC-Suche durchführen
items_training <- s |>
  stac_search(collections = "sentinel-s2-l2a-cogs",
              bbox = c(bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]), 
              datetime = "2021-06-01/2021-06-15") |>
  post_request() |> items_fetch(progress = FALSE)

length(items_training$features)
s2_collection_training <- stac_image_collection(items_training$features, asset_names = assets, property_filter = function(x) {x[["eo:cloud_cover"]] < 20})
s2_collection_training
print(s2_collection_training)

aot_cube <- cube_view(extent = s2_collection_training, srs = "EPSG:25832", dx = 10, dy = 10, dt = "P1M",
                      aggregation = "median", resampling = "average")

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



s <- stac("https://earth-search.aws.element84.com/v0")

items_training_aoi <- s |>
  stac_search(collections = "sentinel-s2-l2a-cogs",
              bbox = c(aoi_target["xmin"], aoi_target["ymin"], aoi_target["xmax"], aoi_target["ymax"]), 
              datetime = "2021-06-01/2021-06-15") |>
  post_request() |> items_fetch(progress = FALSE)

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




# Load the CSV files. These have been created by ChatGPT and should add further arbitrary attributes to our geometry.
# We have the geometry once as a list and once as GeoJSON for csv_data_01. 
# Our function is able to handle both types in order to filter the geometry. 
csv_data_01 <- read_csv("./train_data/train_dat_extended.csv")
csv_data <- read_csv("./train_data/train_dat_geom.csv")




# Function for training a linear regression model. Currently with a time step
lineare_regression_train <- function(aot_cube, csv_data, target_variable, features) {
  # Function for converting string coordinates into sf polygon objects
  convert_to_sf <- function(geom_str) {
    coords <- fromJSON(geom_str)
    if (!all(coords[1, ] == coords[nrow(coords), ])) {
      coords <- rbind(coords, coords[1, ])
    }
    polygon <- st_polygon(list(coords))
    return(polygon)
  }
  
  # Check and convert the geometry column
  if (!inherits(csv_data$geometry, "sfc")) {
    if (grepl("^\\[\\[", csv_data$geometry[1])) {
      # If the geometry is in list format
      message("Convert the geometry column from string to sf...")
      csv_data$geometry <- st_sfc(lapply(csv_data$geometry, convert_to_sf), crs = 4326)
      message("Geometry converted as sf objects")
    } else {
      # If the geometry is available in GeoJSON format
      csv_data$geometry <- st_as_sfc(csv_data$geometry, crs = 4326, GeoJSON = TRUE)
      message("Geometry converted as GeoJSON")
    }
  }
  
  # Ensure that the `fid` column is numeric
  csv_data$fid <- as.integer(csv_data$fid)
  # data frame
  csv_data <- st_as_sf(csv_data)

  # Check whether the conversion was successful
  if (!inherits(csv_data, "sf")) {
    stop("The conversion to an sf object has failed.")
  }
  
  # Extract data from the data cube
  extracted_data <- extract_geom(
    cube = aot_cube,
    sf = csv_data
  )
  message("Trainingsdata extracted ....")
  
  # Ensure that the column names match
  csv_data$ID <- seq_len(nrow(csv_data))

  if (is.list(extracted_data)) {
    message("list extracted_data")
  }
  if (is.list(csv_data)) {
    message("Liste_csv_data")
  }
  
  # Merge the extracted data with the CSV data
  extracted_data <- merge(extracted_data, csv_data, by.x = "FID", by.y = "ID")

  # Remove geometry columns and ensure complete cases
  non_geometry_columns <- !sapply(extracted_data, inherits, "sfc")
  combined_data <- extracted_data[complete.cases(extracted_data[, non_geometry_columns]), ]
  message("Combined data:")
  print(head(combined_data))
  
  # Combine data
  x <- combined_data[, features]
  y <- combined_data[, target_variable]
  
  # Train linear regression model
  message("Training the linear regression model....")
  model_ln <- caret::train(
    x, y,
    method = "lm"
  )
  message("Model trained")
  print(summary(model_ln))
  return(model_ln)
}


# Training the linear regression model with two different data sets. Target variable is the temperature selected.
model_ln <-lineare_regression_train(data_cube_training, csv_data, "temperature", c("B02", "B03", "B04", "B08"))
model_ln_01 <-lineare_regression_train(data_cube_training, csv_data_01, "temperature", c("B02", "B03", "B04", "B08"))

#############Deep Learning############
# Define the TempCNN model
# https://cran.r-project.org/web/packages/keras/vignettes/sequential_model.html
temp_cnn_model <- function(input_shape, num_classes) {
  model <- keras_model_sequential() %>%
    layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu', input_shape = input_shape) %>%
    layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu') %>%
    layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu') %>%
    layer_flatten() %>%
    layer_dense(units = 256, activation = 'relu') %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = num_classes, activation = 'softmax')
  
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_adam(),
    metrics = c('accuracy')
  )
  
  return(model)
}


# Function for preparing the data and training the TempCNN. Currently with a time step. 
train_model_temp_cnn <- function(aot_cube, train_dat) {
  cube_crs <- gdalcubes::srs(aot_cube)
  crs_data <- as.numeric(gsub("EPSG:", "", cube_crs))
  
  # Transform the training data into the CRS of the data cube
  train_dat <- sf::st_transform(train_dat, crs = crs_data)
  print(train_dat)
  
  message("Combining training data with cube data . . . .")
  
  # Extract data from the data cube based on the geometries
  extraction <- extract_geom(
    cube = aot_cube,
    sf = train_dat
  )
  message("Trainingsdata extracted ....")
  
  if (nrow(extraction) == 0) {
    stop("No data extracted. Check if the bounding boxes of the training data and the data cube overlap.")
  }
  print(extraction)
  
  # Merge training data with extracted data
  message("Now merging trainingsdata with aoi data ....")
  train_dat$PolyID <- seq_len(nrow(train_dat))
  print(train_dat)
  extraction <- merge(extraction, train_dat, by.x = "FID", by.y = "PolyID")
  print(extraction)
  message("Extraction merged with trainingsdata ....")
  
  # Prepare training data for model training
  message("Now preparing the trainingdata for the modeltraining ....")
  predictors <- c("B02", "B03", "B04", "B08") 
  train_ids <- createDataPartition(extraction$FID, p = 0.8, list = FALSE)
  train_data <- extraction[train_ids, ]
  test_data <- extraction[-train_ids, ]
  
  train_data <- train_data[complete.cases(train_data[, predictors]), ]
  test_data <- test_data[complete.cases(test_data[, predictors]), ]
  
  # Convert to numerical data
  train_data <- as.data.frame(train_data)
  test_data <- as.data.frame(test_data)
  
  x_train <- as.matrix(train_data[, predictors])
  y_train <- as.numeric(as.factor(train_data$Label)) - 1
  
  x_test <- as.matrix(test_data[, predictors])
  y_test <- as.numeric(as.factor(test_data$Label)) - 1
  
  # Reshape the Data for the CNN (Samples, Timesteps, Features)
  num_timesteps <- 1
  x_train <- array_reshape(x_train, c(nrow(x_train), num_timesteps, length(predictors)))
  x_test <- array_reshape(x_test, c(nrow(x_test), num_timesteps, length(predictors)))
  
  # One-Hot Encoding target Label
  num_classes <- length(unique(y_train))
  y_train <- to_categorical(y_train, num_classes = num_classes)
  y_test <- to_categorical(y_test, num_classes = num_classes)
  
  # Define and train model
  message("model is now being trained")
  model <- temp_cnn_model(input_shape = c(num_timesteps, length(predictors)), num_classes = num_classes)
  
  history <- model %>% fit(
    x_train, y_train,
    epochs = 20, batch_size = 32,
    validation_data = list(x_test, y_test)
  )
  
  # Evaluate model
  score <- model %>% evaluate(x_test, y_test)
  message("Model trained, accuracy: ", score[2])  
  
  return(model)
}

model_cnn <- train_model_temp_cnn(data_cube_training, training_sites)
model_cnn

# save model the model  as an h5.file
save_model_hdf5(model_cnn, 'cnn_model.h5')

# Function for extracting data from the gdalcubes cube
extract_cube_data <- function(cube) {
  # Create a temporary file to save the NetCDF data
  temp_file <- tempfile(fileext = ".nc")
  
  # NetCDF file for data cube
  gdalcubes::write_ncdf(cube, temp_file)
  
  # Read the NetCDF file as raster data
  raster_data <- terra::rast(temp_file)
  
  # Convert the raster data into a matrix
  matrix_data <- as.matrix(raster_data)
  
  # Return of the matrix data, the extent, the coordinate system and the dimensions of the grid
  return(list(matrix_data = matrix_data, extent = ext(raster_data), crs = crs(raster_data), dim = dim(raster_data)))
}

# Customized prediction function for CNN models
prediction_for_cnn <- function(model_path, aoi_cube) {
  tryCatch({
    #load the model
    model <- load_model_hdf5(model_path)
    
    # Extract data
    extracted_data <- extract_cube_data(aoi_cube)
    x_data <- extracted_data$matrix_data
    raster_extent <- extracted_data$extent
    raster_crs <- extracted_data$crs
    original_dim <- extracted_data$dim
    
    # Ensure that the input data has the form (batch_size, timesteps, features)
    x_data <- array_reshape(x_data, c(dim(x_data)[1], 1, dim(x_data)[2]))
    
    # Perform predictions
    predictions <- predict(model, x_data)
    predictions <- apply(predictions, 1, which.max) - 1 
    message("Prediction calculated ....")
    
    
    # Convert the predictions back to the original grid form
    prediction_matrix <- matrix(predictions, nrow = original_dim[1], ncol = original_dim[2], byrow = TRUE)
    
    # Temporary file for saving the predictions
    temp_file <- tempfile(fileext = ".tif")
    
    # Create a raster object from the predictions
    prediction_raster <- terra::rast(prediction_matrix, extent = raster_extent, crs = raster_crs)
    
    # Save the predictions as TIFF
    terra::writeRaster(prediction_raster, temp_file, filetype = "GTiff")
    
    # Read the saved file as a raster
    prediction_raster <- terra::rast(temp_file)
    
    return(prediction_raster)
  },
  error = function(e){
    message("Error in classification")
    message(conditionMessage(e))
  })
}


# Prediction function for the Linear Regression
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
prediction_ln <- prediction(model = model_ln, aoi_cube = aoi_training)
prediction_ln_01 <- prediction(model = model_ln_01, aoi_cube = aoi_training)

prediction_raster_ln <- terra::rast(gdalcubes::write_tif(prediction_ln))
prediction_raster_ln_01 <- terra::rast(gdalcubes::write_tif(prediction_ln_01))

# Performs the prediction with the model and the aoi-cube for the TempCnn
prediction_raster_cnn <- prediction_for_cnn('cnn_model.h5', aoi_training)
plot(prediction_raster_cnn)



plot(prediction_raster_ln)
plot(prediction_raster_ln_01)
