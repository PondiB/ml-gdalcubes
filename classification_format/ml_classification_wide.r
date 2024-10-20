library(gdalcubes)
library(sf)
library(rstac)
library(randomForest)
library(caret)
library(xgboost)
library(dplyr)
library(readr)
library(terra)
library(jsonlite)
library(stats)
library(kernlab)
library(reticulate)
py_install(c("onnxmltools", "skl2onnx", "scikit-learn", "xgboost"))



# Extracts the coordinate system (CRS) of the data cube
extract_crs_datacube <- function(aot_cube){
  cube_crs <- gdalcubes::srs(aot_cube)
  crs_data <- as.numeric(gsub("EPSG:", "", cube_crs))
  print(paste("CRS of the data cube:", crs_data))
  return(crs_data)
}

# Transforms the training data into the coordinate system of the data cube  
transform_training_data <- function(train_dat, aot_crs){
  train_dat <- sf::st_transform(train_dat, crs = aot_crs)
  print("Training data transformed:")
  print(train_dat)
  return(train_dat)
}


################################################################################
data_preprocessing_single <- function(aot_cube, train_dat){
  print("Starting data extraction...")
  extraction <- gdalcubes::extract_geom(
    cube = aot_cube,
    sf = train_dat
  )
  print("Extraction result:")
  print(extraction)
  
  if (nrow(extraction) == 0) {
    stop("No data extracted. Check if the bounding boxes of the training data and the data cube overlap.")
  }
  predictors_name <- get_cube_band_names(aot_cube)
  train_dat$PolyID <- seq_len(nrow(train_dat))
  extraction <- base::merge(extraction, train_dat, by.x = "FID", by.y = "PolyID")
  message("Extraction merged with training data ....")
  
  train_ids <- caret::createDataPartition(extraction$FID, p = 0.2, list = FALSE)
  train_data <- extraction[train_ids, ]
  train_data <- train_data[stats::complete.cases(train_data[, predictors_name]), ]
  train_data <- base::as.data.frame(train_data)
  message("Training data prepared . . . .")
  return(train_data)
}

data_preprocessing_multiple <- function(aot_cube, train_dat) {
  print("Starting data extraction for multiple time steps...")
  
  extraction <- extraction1
  
  unique_times <- unique(extraction$time)
  print(paste("You have", length(unique_times), "time steps:"))
  print(unique_times)
  
  if (nrow(extraction) == 0) {
    stop("No data extracted. Check if the bounding boxes of the training data and the data cube overlap.")
  }
  
  train_dat$PolyID <- seq_len(nrow(train_dat))
  extraction <- merge(extraction, train_dat, by.x = "FID", by.y = "PolyID", all.x = TRUE)
  extraction$pixel_id <- seq_len(nrow(extraction))
  extraction$time_numeric <- as.numeric(as.Date(extraction$time))
  
  return(extraction)
}

# Konvertiert die Daten ins Wide-Format
convert_to_wide_format <- function(train_data, band_names) {
  time_steps <- unique(train_data$time)
  n_steps <- length(time_steps)
  
  data_frames <- list()
  
  # Iteriere durch die Zeitschritte und speichere die Datenframes in der Liste
  for (i in seq_along(time_steps)) {
    data_time <- train_data %>%
      filter(time == time_steps[i]) %>%
      select(-geometry) %>%  
      rename_with(~ paste0(., "_T", i), all_of(band_names)) %>%  
      select(FID, pixel_id, starts_with("B"), -time, -time_numeric)  
    
    # Weise eine konsistente Pixel-ID zu
    data_time$pixel_id <- seq_len(nrow(data_time))
    
    data_frames[[i]] <- data_time
    print(paste("Daten für Zeitschritt", i, "umbenannt:"))
    print(data_time)
  }
  
  wide_data <- Reduce(function(x, y) inner_join(x, y, by = c("FID", "pixel_id")), data_frames)
  
  complete_data <- wide_data[complete.cases(wide_data), ]
  
  print("Daten im Wide-Format nach dem Zusammenfügen der Zeitschritte:")
  print(complete_data)
  
  return(complete_data)
}

# Extrahiert die Bandnamen des Datacubes
get_cube_band_names <- function(cube) {
  band_names <- names(cube)
  return(band_names)
}

#############################################################################
## algorithm definitions

# Trainiert ein Random-Forest-Modell (für einen oder mehrere Zeitschritte)
ml_fit_class_random_forest <- function(aot_cube, training_set, target_column, num_trees = 100, max_variables = NULL, seed = NULL, multiple_time_steps = FALSE) {
  
  crs_data <- extract_crs_datacube(aot_cube = aot_cube)
  train_dat <- transform_training_data(train_dat = training_set, aot_crs = crs_data)
  
  if (multiple_time_steps) {
    train_data <- data_preprocessing_multiple(aot_cube = aot_cube, train_dat = train_dat)
    band_names <- get_cube_band_names(aot_cube)
    train_data_wide <- convert_to_wide_format(train_data = train_data, band_names = band_names)
    
    labels <- training_set %>%
      st_set_geometry(NULL) %>%
      distinct(fid, Label)
    train_data <- merge(train_data_wide, labels, by.x = "FID", by.y = "fid")
    
    # Prädiktoren für mehrere Zeitschritte
    predictor_names <- setdiff(names(train_data), c("pixel_id", "FID", target_column, "geometry"))
    
  } else {
    train_data <- data_preprocessing_single(aot_cube = aot_cube, train_dat = train_dat)
    
    # Prädiktoren für einen einzelnen Zeitschritt
    predictor_names <- get_cube_band_names(aot_cube)
  }
  
  y <- as.factor(train_data[[target_column]])
  x <- as.data.frame(lapply(train_data[, predictor_names], as.numeric))
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  if (is.null(max_variables)) {
    max_variables <- floor(sqrt(ncol(x)))
  }
  
  model <- randomForest::randomForest(
    x = x,
    y = y,
    ntree = num_trees,
    mtry = max_variables,
    importance = TRUE
  )
  
  print(model)
  return(list(model = model, train_data = train_data))
}


################################################################################



ml_fit_class_svm <- function(aot_cube, training_set, target_column, 
                             kernel = "radial", C = 1, sigma = 1, 
                             gamma = NULL, degree = 3, coef0 = 0, seed = NULL, multiple_time_steps = FALSE) {
  
  # Extrahiere das CRS des Datacubes
  crs_data <- extract_crs_datacube(aot_cube = aot_cube)
  
  # Transformiere die Trainingsdaten ins CRS des Datacubes
  train_dat <- transform_training_data(train_dat = training_set, aot_crs = crs_data)
  
  if (multiple_time_steps) {
    # Datenvorverarbeitung für mehrere Zeitschritte
    train_data <- data_preprocessing_multiple(aot_cube = aot_cube, train_dat = train_dat)
    band_names <- get_cube_band_names(aot_cube)
    train_data_wide <- convert_to_wide_format(train_data = train_data, band_names = band_names)
    
    # Füge die Zielvariable (Label) hinzu
    labels <- training_set %>%
      st_set_geometry(NULL) %>%
      distinct(fid, Label)
    train_data <- merge(train_data_wide, labels, by.x = "FID", by.y = "fid")
    
    # Prädiktoren für mehrere Zeitschritte
    predictor_names <- setdiff(names(train_data), c("pixel_id", "FID", target_column, "geometry"))
    
  } else {
    # Datenvorverarbeitung für einen einzelnen Zeitschritt
    train_data <- data_preprocessing_single(aot_cube = aot_cube, train_dat = train_dat)
    
    # Prädiktoren für einen Zeitschritt
    predictor_names <- get_cube_band_names(aot_cube)
  }
  
  x <- as.data.frame(lapply(train_data[, predictor_names], as.numeric))
  y <- as.factor(train_data[[target_column]])
  
  # Setze den Seed, falls angegeben
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # Wähle das passende SVM-Modell basierend auf dem Kernel-Typ
  if (kernel == "radial") {
    tune_grid <- expand.grid(C = C, sigma = sigma)
    method <- "svmRadial"
  } else if (kernel == "linear") {
    tune_grid <- expand.grid(C = C)
    method <- "svmLinear"
  } else if (kernel == "polynomial") {
    tune_grid <- expand.grid(C = C, degree = degree, scale = gamma, coef0 = coef0)
    method <- "svmPoly"
  } else {
    stop("Unsupported kernel type. Choose from 'radial', 'linear', or 'polynomial'.")
  }
  
  # Cross-Validation-Setup
  train_control <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 3)
  
  # Trainiere das SVM-Modell
  svm_model <- caret::train(
    x = x, 
    y = y,
    method = method,
    tuneGrid = tune_grid,
    trControl = train_control,
    preProcess = c("center", "scale")
  )
  
  message("SVM Model trained successfully.")
  
  # Gebe das Modell und die Trainingsdaten zurück
  return(list(model = svm_model, train_data = train_data))
}




###################################################################
ml_fit_class_xgboost <- function(aot_cube, training_set, target_column, 
                                 learning_rate = 0.15, max_depth = 5, min_child_weight = 1, 
                                 subsample = 0.8, colsample_bytree = 1, min_split_loss = 1, 
                                 nrounds = 100, seed = NULL, multiple_time_steps = TRUE) {
  
  # Extrahiere das CRS des Datacubes
  crs_data <- extract_crs_datacube(aot_cube = aot_cube)
  
  # Transformiere die Trainingsdaten ins CRS des Datacubes
  train_dat <- transform_training_data(train_dat = training_set, aot_crs = crs_data)
  
  if (multiple_time_steps) {
    train_data <- data_preprocessing_multiple(aot_cube = aot_cube, train_dat = train_dat)
    band_names <- get_cube_band_names(aot_cube)
    train_data_wide <- convert_to_wide_format(train_data = train_data, band_names = band_names)
    
    # Füge die Zielvariable (Label) hinzu
    labels <- training_set %>%
      st_set_geometry(NULL) %>%
      distinct(fid, Label)
    train_data <- merge(train_data_wide, labels, by.x = "FID", by.y = "fid")
    
    # Prädiktoren für mehrere Zeitschritte
    predictor_names <- setdiff(names(train_data), c("pixel_id", "FID", target_column, "geometry"))
    
  } else {
    # Datenvorverarbeitung für einen einzelnen Zeitschritt
    train_data <- data_preprocessing_single(aot_cube = aot_cube, train_dat = train_dat)
    
    # Prädiktoren für einen Zeitschritt
    predictor_names <- get_cube_band_names(aot_cube)
  }
  
  x <- as.data.frame(lapply(train_data[, predictor_names], as.numeric))
  y <- as.factor(train_data[[target_column]])
  
  # Setze den Seed, falls angegeben
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # Sets the parameters for the xgboost model
  tune_grid <- expand.grid(
    nrounds = nrounds,
    max_depth = max_depth,
    eta = learning_rate,
    gamma = min_split_loss,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    subsample = subsample
  )
  
  # Trainiere das XGBoost-Modell
  model <- caret::train(
    x = x,
    y = y,
    method = "xgbTree",
    trControl = trainControl(method = "cv", number = 5, search = "grid"),
    tuneGrid = tune_grid
  )
  
  message("XGBoost Model trained successfully.")
  
  # Gebe das Modell und die Trainingsdaten zurück
  return(list(model = model, train_data = train_data))
}


###################################################################
## Prediction model single
ml_predict <- function(data_cube, model, dimension = NULL) {
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

## Prediction model multi

time_steps_query <- function(cube){
  time_steps <- gdalcubes::dimension_values(aoi_cube_multi)
  time_steps <- time_steps$t
  return(time_steps)
}


prepare_and_combine_multitemporal_cube <- function(object, select_bands, time_steps) {
  data_frames <- list()
  
  # Iteriere über jeden Zeitschritt
  for (i in seq_along(time_steps)) {
    # Extrahiere die Werte für den aktuellen Zeitschritt
    cube <- gdalcubes::select_bands(object, select_bands)
    cube <- gdalcubes::slice_time(cube, time_steps[i])
    
    rename_map <- setNames(paste0(select_bands, "_T", i), select_bands)
    cube <- do.call(gdalcubes::rename_bands, c(list(cube), rename_map))
    
    # Entferne die Zeitdimension und konvertiere den Cube zu einem DataFrame
    data_frame <- as.data.frame(gdalcubes::reduce_time(cube, FUN = function(data) { return(data) }))
    
    data_frames[[i]] <- data_frame
  }
  
  # Kombiniere alle DataFrames entlang der Spalten
  combined_data <- do.call(cbind, data_frames)
  
  return(combined_data)
}


clean_and_predict <- function(data_frame, new_names, model) {
  # Entferne Spalten mit fehlenden Werten
  cleaned_data <- data_frame[, colSums(is.na(data_frame)) == 0]
  
  if (length(new_names) == ncol(cleaned_data)) {
    names(cleaned_data) <- new_names
  } else {
    stop("Die Anzahl der neuen Spaltennamen stimmt nicht mit der Anzahl der Spalten im bereinigten DataFrame überein.")
  }
  
  predictions <- predict(model, cleaned_data)
  
  return(predictions)
}




# Funktion, um ein Raster aus einem gdalcubes Daten-Cube basierend auf den Dimensionen zu erstellen
create_raster_from_cube <- function(cube, predictions, srs = NULL) {
  dims <- dimensions(cube)
  
  nx <- dims$x$count
  ny <- dims$y$count
  
  rast_template <- terra::rast(ncols = nx, nrows = ny)
  
  terra::ext(rast_template) <- terra::ext(dims$x$low, dims$x$high, dims$y$low, dims$y$high)
  
  if (!is.null(srs)) {
    terra::crs(rast_template) <- srs
  } else {
    stop("Bitte einen gültigen 'srs' Parameter angeben")
  }
  
  prediction_raster <- rast_template
  values(prediction_raster) <- as.vector(predictions)
  
  return(prediction_raster)
}




ml_predict_multi <- function(cube, select_bands, time_steps, model, new_names, srs = NULL) {
  combined_data <- prepare_and_combine_multitemporal_cube(cube, select_bands, time_steps)
  
  predictions <- clean_and_predict(combined_data, new_names, model)
  
  prediction_raster <- create_raster_from_cube(cube, predictions, srs)
  
  return(prediction_raster)
}


###################################################################



save_model_as_rds <- function(model, filepath) {
  tryCatch({
    saveRDS(model, file = filepath)
    message(paste("Model saved successfully to", filepath))
  }, error = function(e) {
    message("Error in saving the model as RDS.")
    message(conditionMessage(e))
  })
}


check_train_data <- function(trainings_data){
  # Bereinige die Trainingsdaten
  train_data_clean <- trainings_data
  train_data_clean[predictors] <- lapply(train_data_clean[predictors], function(x) as.numeric(as.character(x)))
  train_data_clean <- na.omit(train_data_clean) 
  return(train_data_clean)
}

conv_numeric <- function(train_data_clean){
  train_data_clean$Label <- as.numeric(as.factor(train_data_clean$Label)) - 1  
  return(train_data_clean)
}



save_model_as_onnx <- function(model, model_type, filepath, predictors, train_data) {
  library(reticulate)
  
  # Importiere notwendige Python-Module
  onnxmltools <- import("onnxmltools")
  skl2onnx <- import("skl2onnx")
  xgboost <- import("xgboost")
  sklearn <- import("sklearn.ensemble")
  sklearn_svm <- import("sklearn.svm")  
  onnx <- import("onnx")
  np <- import("numpy")
  
  # Bereinige die Trainingsdaten
  train_data_clean<-check_train_data(train_data)
  # Konvertiere Labels zu Faktoren und dann zu numerischen Werten
  train_data_clean <- conv_numeric(train_data_clean = train_data_clean)
  
  # Konvertiere die bereinigten Trainingsdaten zu numpy arrays
  x_train <- np$array(as.matrix(train_data_clean[, predictors]))
  y_train <- np$array(as.integer(train_data_clean$Label))
  
  # Definiere den Eingabedatentyp für ONNX
  FloatTensorType <- reticulate::import("skl2onnx.common.data_types")$FloatTensorType
  initial_type <- list(list("float_input", FloatTensorType(list(NULL, length(predictors)))))
  
  # Modellkonvertierung basierend auf dem Modelltyp
  if (model_type == "xgboost") {
    # Lade das zuvor gespeicherte native XGBoost-Modell
    xgb_model_py <- xgboost$XGBClassifier()
    xgb_model_py$load_model("xgboost_native_model.bin")
    
    # Konvertiere das XGBoost-Modell zu ONNX
    onnx_model <- onnxmltools$convert_xgboost(xgb_model_py, initial_types = initial_type)
    
  } else if (model_type == "random_forest") {
    rf_model_py <- sklearn$RandomForestClassifier(n_estimators = as.integer(100))
    rf_model_py$fit(x_train, y_train)  # Fit mit bereinigten und konvertierten Daten
    onnx_model <- onnxmltools$convert_sklearn(rf_model_py, initial_types = initial_type)
    
  } else if (model_type == "svm") {
    svm_model_py <- sklearn_svm$SVC(kernel = "rbf", C = 1.0)
    svm_model_py$fit(x_train, y_train)  # Fit mit bereinigten und konvertierten Daten
    onnx_model <- skl2onnx$convert_sklearn(svm_model_py, initial_types = initial_type)
  }
  
  # Speichere das ONNX-Modell
  onnx$save_model(onnx_model, filepath)
  message(paste("Modell erfolgreich als ONNX gespeichert unter:", filepath))
}








# non openeo functions for training data preprocessing
extract_values <- function(bbox, crop_box, assets){
  # STAC-Server URL
  s <- rstac::stac("https://earth-search.aws.element84.com/v0")
  
  # Perform STAC search
  items_training <- s |>
    rstac::stac_search(collections = "sentinel-s2-l2a-cogs",
                       bbox = c(bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]), 
                       datetime = "2021-06-01/2021-06-15") |>
    rstac::post_request() |> rstac::items_fetch(progress = FALSE)
  
  # Number of elements found
  length(items_training$features)
  s2_collection_training <- gdalcubes::stac_image_collection(items_training$features, asset_names = assets, property_filter = function(x) {x[["eo:cloud_cover"]] < 20})
  s2_collection_training
  print(s2_collection_training)
  
  
  # Define the bounding box based on the actual values of the data cube
  cube <- gdalcubes::cube_view(extent = s2_collection_training, srs = "EPSG:25832", dx = 10, dy = 10, dt = "P1M",
                               aggregation = "median", resampling = "average")
  
  
  # Create the data cube with the defined view
  data_cube <- gdalcubes::raster_cube(s2_collection_training, cube) %>%
    gdalcubes::select_bands(c("B02", "B03", "B04", "B08")) %>%
    gdalcubes::crop(extent = list(left = crop_box["xmin"],
                                  right = crop_box["xmax"],
                                  bottom = crop_box["ymin"],
                                  top = crop_box["ymax"],
                                  t0 = "2021-06", t1 = "2021-06"),
                    snap = "near")
  data_cube
  return(data_cube)
}

extract_values_multi <- function(bbox, crop_box, assets){
  # STAC-Server URL
  s <- rstac::stac("https://earth-search.aws.element84.com/v0")
  
  # Perform STAC search
  items_training <- s |>
    rstac::stac_search(collections = "sentinel-s2-l2a-cogs",
                       bbox = c(bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]), 
                       datetime = "2021-01-01/2021-12-31") |>  # Full year 2021
    rstac::post_request() |> rstac::items_fetch(progress = FALSE)
  
  # Number of elements found
  length(items_training$features)
  s2_collection_training <- gdalcubes::stac_image_collection(items_training$features, asset_names = assets, property_filter = function(x) {x[["eo:cloud_cover"]] < 20})
  s2_collection_training
  print(s2_collection_training)
  
  # Define the bounding box based on the actual values of the data cube
  cube <- gdalcubes::cube_view(extent = s2_collection_training, srs = "EPSG:25832", dx = 10, dy = 10, dt = "P1M",  # Monthly resolution
                               aggregation = "median", resampling = "average")
  
  # Create the data cube with the defined view
  data_cube <- gdalcubes::raster_cube(s2_collection_training, cube) %>%
    gdalcubes::select_bands(c("B02", "B03", "B04", "B08")) %>%
    gdalcubes::crop(extent = list(left = crop_box["xmin"],
                                  right = crop_box["xmax"],
                                  bottom = crop_box["ymin"],
                                  top = crop_box["ymax"],
                                  t0 = "2021-06", t1 = "2021-11"),
                    snap = "near")
  
  # Optional: Apply temporal aggregation, e.g., aggregate every 3 months
  aggregated_data_cube <- gdalcubes::aggregate_time(data_cube, dt = "P3M", method = "mean")  # Example: Quarterly aggregation
  
  return(aggregated_data_cube)
}







training_sites <- sf::read_sf("./train_data/train_dat.geojson")
assets <- c("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "SCL")

bbox <- sf::st_bbox(training_sites)
# Define the bounding box for the training area
aot_test <- sf::st_bbox(c(xmin = 388831.6, ymin = 5698900.1, xmax = 398063.3, ymax = 5718133.7), crs = sf::st_crs(25832))

# Define the bounding box in EPSG:4326 The coordinates form a bounding box in which Paderborn is located
aoi_target <- sf::st_bbox(c(xmin = 8.6638, ymin = 51.6612, xmax = 8.8410, ymax = 51.7815), crs = sf::st_crs(4326))

# Conversion to an sf object
bbox_target_sf <- sf::st_as_sfc(aoi_target)

# Transform the coordinates to EPSG:25832
bbox_target_transformed <- sf::st_transform(bbox_target_sf, crs = sf::st_crs(25832))

# Extract the transformed coordinates
bbox_target_utm <- sf::st_bbox(bbox_target_transformed)

predictors <- get_cube_band_names(aot_cube)
print(predictors)


# cube initialization
aot_cube<- extract_values(bbox = bbox, crop_box = aot_test, assets = assets )
aoi_cube <- extract_values(bbox = aoi_target, crop_box = bbox_target_utm, assets = assets)
aot_cube_multi<- extract_values_multi(bbox = bbox, crop_box = aot_test, assets = assets )
aoi_cube_multi <- extract_values_multi(bbox = aoi_target, crop_box = bbox_target_utm, assets = assets)
###################################
## function call

### Random Forest single
rf_result_single <- ml_fit_class_random_forest(aot_cube = aot_cube, training_set = training_sites, target_column = "Label", multiple_time_steps = FALSE)
random_forest_model <- rf_result_single$model
train_data_rf <- rf_result_single$train_data


r_ml <- ml_predict(data_cube = aoi_cube, model = random_forest_model)
prediction_raster_rf <- terra::rast(gdalcubes::write_tif(r_ml))
save_model_as_rds(model = random_forest_model, filepath = "random_forest_model.rds")
save_model_as_onnx(random_forest_model, "random_forest", "random_forest_model.onnx", select_bands, train_data_rf)

plot(prediction_raster_rf)

### Random Forest multi
rf_result_multi <- ml_fit_class_random_forest(aot_cube = aot_cube_multi, training_set = training_sites, target_column = "Label", multiple_time_steps = TRUE)
rf_result_multi_model <- rf_result_multi$model
rf_result_multi_train_data <- rf_result_multi$train_data

# Beispielanwendung der Funktion
select_bands <- get_cube_band_names(aoi_cube_multi)
time_steps <- time_steps_query(cube = aoi_cube_multi)
time_steps
new_names <- c("B02_T1", "B03_T1", "B04_T1", "B08_T1", "B02_T2", "B03_T2", "B04_T2", "B08_T2")

prediction_raster_multi <- ml_predict_multi(
  cube = aoi_cube_multi,
  select_bands = select_bands,
  time_steps = time_steps,
  model = rf_result_multi_model,  
  new_names = new_names,
  srs = "EPSG:25832"
)

plot(prediction_raster_multi)



### SVM single
svm_result_single <- ml_fit_class_svm(aot_cube = aot_cube, training_set = training_sites, 
                                      target_column = "Label", kernel = "radial", 
                                      C = 1, sigma = 0.5, multiple_time_steps = FALSE)
svm_model_single <- svm_result_single$model
svm_train_data_single <- svm_result_single$train_data
svm_predict_single <- ml_predict(data_cube = aoi_cube, model = svm_model_single)
prediction_raster_svm <- terra::rast(gdalcubes::write_tif(svm_predict_single))
plot(prediction_raster_svm)
save_model_as_rds(model = svm_model_single, filepath = "svm_model.rds")
save_model_as_onnx(svm_model_single, "svm", "svm.onnx", select_bands, svm_train_data_single)





## SVM mulit
# Beispielanwendung der Funktion für einen einzelnen oder mehrere Zeitschritte
svm_result_multi <- ml_fit_class_svm(aot_cube = aot_cube_multi, training_set = training_sites, 
                               target_column = "Label", kernel = "radial", 
                               C = 1, sigma = 0.5, multiple_time_steps = TRUE)
svm_model_multi <- svm_result_multi$model
svm_train_data <- svm_result_multi$train_data
select_bands <- get_cube_band_names(aoi_cube_multi)
time_steps <- time_steps_query(cube = aoi_cube_multi)
time_steps
new_names <- c("B02_T1", "B03_T1", "B04_T1", "B08_T1", "B02_T2", "B03_T2", "B04_T2", "B08_T2")

# Aufruf der neuen Funktion mit den gewünschten Parametern
prediction_raster_multi <- ml_predict_multi(
  cube = aoi_cube_multi,
  select_bands = select_bands,
  time_steps = time_steps,
  model = svm_model_multi,  
  new_names = new_names,
  srs = "EPSG:25832"
)

# Anzeige des resultierenden Rasters
plot(prediction_raster_multi)




### xgboost single

xgboost_result_single <- ml_fit_class_xgboost(aot_cube = aot_cube, training_set = training_sites, 
                                              target_column = "Label", learning_rate = 0.1, max_depth = 6, 
                                              min_child_weight = 1, subsample = 0.8, colsample_bytree = 1, 
                                              nrounds = 150, multiple_time_steps = FALSE)

xgboost_model_single <- xgboost_result_single$model
xgboost_train_data_single <- xgboost_result_single$train_data
xgboost_predict_single <- ml_predict(data_cube = aoi_cube, model = xgboost_model_single)
prediction_raster_xgboost <- terra::rast(gdalcubes::write_tif(xgboost_predict_single))
plot(prediction_raster_xgboost)
xgboost::xgb.save(xgboost_model_single$finalModel, "xgboost_native_model.bin")
save_model_as_rds(model = xgboost_model_single, filepath = "xgboost_model.rds")
save_model_as_onnx(xgboost_model_single, "xgboost", "xgboost_model.onnx", select_bands, xgboost_train_data_single)



### xgboost multi
xgboost_result <- ml_fit_class_xgboost(aot_cube = aot_cube_multi, training_set = training_sites, 
                                       target_column = "Label", learning_rate = 0.1, max_depth = 6, 
                                       min_child_weight = 1, subsample = 0.8, colsample_bytree = 1, 
                                       nrounds = 150, multiple_time_steps = TRUE)
xgboost_model_multi <- xgboost_result$model
xgboost_train_data_multi <- xgboost_result$train_data

select_bands <- get_cube_band_names(aoi_cube_multi)
time_steps <- time_steps_query(cube = aoi_cube_multi)
time_steps
new_names <- c("B02_T1", "B03_T1", "B04_T1", "B08_T1", "B02_T2", "B03_T2", "B04_T2", "B08_T2")

# Aufruf der neuen Funktion mit den gewünschten Parametern
prediction_raster_multi <- ml_predict_multi(
  cube = aoi_cube_multi,
  select_bands = select_bands,
  time_steps = time_steps,
  model = xgboost_model_multi,  
  new_names = new_names,
  srs = "EPSG:25832"
)

# Anzeige des resultierenden Rasters
plot(prediction_raster_multi)



#####onnx####
load_and_check_onnx_model <- function(model_path) {
  # Importieren Sie die notwendigen Python-Bibliotheken
  onnx <- reticulate::import("onnx", convert = TRUE)
  onnxruntime <- reticulate::import("onnxruntime", convert = TRUE)
  
  # Versuchen Sie, die ONNX-Modelldatei zu laden
  tryCatch({
    onnx_model <- onnx$load(model_path)
    print("Das ONNX-Modell wurde erfolgreich geladen.")
  }, error = function(e) {
    print("Fehler beim Laden des ONNX-Modells:")
    print(e)
  })
  
  # Erstellen Sie die Inferenz-Session und fangen Sie Fehler ab
  session <- NULL
  tryCatch({
    session <- onnxruntime$InferenceSession(model_path)
    print("Inference-Session wurde erfolgreich erstellt.")
  }, error = function(e) {
    print("Fehler bei der Erstellung der Inference-Session:")
    print(e)
  })
  
  # Überprüfen Sie, ob die Inference-Session korrekt initialisiert wurde
  if (!is.null(session)) {
    input_list <- session$get_inputs()
    
    # Check if input list has any inputs
    if (length(input_list) > 0) {
      input_name <- input_list[[1]]$name
      print(paste("Der Name des Eingangs ist:", input_name))
      
      # Zeige den ersten Input an
      print(paste("Input Name:", input_list[[1]]$name))
      print(paste("Input Typ:", input_list[[1]]$type))
      print(paste("Input Form:", input_list[[1]]$shape))
    } else {
      print("Keine Eingänge im Modell gefunden.")
    }
  } else {
    print("Inference-Session wurde nicht erfolgreich erstellt.")
  }
  
  # Rückgabe des Modells und der Session, falls erforderlich
  return(list(onnx_model = onnx_model, session = session))
}

# Beispiel für die Verwendung der Funktion
random_forest_model <- load_and_check_onnx_model("random_forest_model.onnx")
svm_model <- load_and_check_onnx_model("svm.onnx")
xgboost_model <- load_and_check_onnx_model("xgboost_model.onnx")

extract_cube <- function(cube, predictors) {
  # Extrahiere die Werte des AOI-Würfels und konvertiere sie in ein R-Array
  aoi_values <- gdalcubes::as_array(cube)
  
  # Überprüfe die Dimensionen des aoi_values
  cube_dims <- dim(aoi_values)
  print(paste("Dimensionen des AOI-Würfels:", cube_dims))
  
  # Reshape input_matrix, um die korrekte Anzahl der Beobachtungen zu haben
  input_matrix <- matrix(aperm(aoi_values, c(3, 4, 1, 2)), ncol = length(predictors))
  
  # Überprüfen Sie die Dimensionen der extrahierten Matrix
  print(paste("Eingabedaten-Matrixdimensionalität:", dim(input_matrix)))
  
  # Konvertiere Matrix in numpy-Array vom Typ float32
  input_matrix <- reticulate::np_array(input_matrix, dtype = "float32")
  
  return(list(dims = cube_dims, data = input_matrix))
}


predict_with_onnx_model <- function(onnx_model_path, aoi_cube, predictors) {
  onnx <- reticulate::import("onnx")
  onnxruntime <- reticulate::import("onnxruntime")
  
  # Lade das ONNX-Modell
  tryCatch({
    session <- onnxruntime$InferenceSession(onnx_model_path)
    message("Inference-Session wurde erfolgreich erstellt.")
  }, error = function(e) {
    stop("Fehler bei der Erstellung der Inference-Session: ", e$message)
  })
  
  # Überprüfe, ob die Inference-Session korrekt initialisiert wurde
  input_name <- session$get_inputs()[[1]]$name
  print(paste("Der Name des Eingangs ist:", input_name))
  
  # Verwende die extract_cube-Funktion und greife auf die Rückgabewerte zu
  extracted_data <- extract_cube(cube = aoi_cube, predictors = predictors)
  input_matrix <- extracted_data$data
  cube_dims <- extracted_data$dims
  
  # Vorhersage mit dem ONNX-Modell durchführen
  input_feed <- list(float_input = input_matrix)
  tryCatch({
    outputs <- session$run(NULL, input_feed)
    message("Vorhersage erfolgreich durchgeführt.")
  }, error = function(e) {
    stop("Fehler bei der Vorhersage: ", e$message)
  })
  
  # Korrigiere die Dimensionen für das Vorhersage-Array
  nrow_cube <- cube_dims[3]  # Höhe
  ncol_cube <- cube_dims[4]  # Breite
  
  # Reshape der Ausgabe
  pred_array <- matrix(outputs[[1]], nrow = nrow_cube, ncol = ncol_cube)
  
  # Konvertiere die Vorhersageausgabe in ein Raster
  prediction_raster <- terra::rast(pred_array, crs = gdalcubes::srs(aoi_cube))
  
  # Überprüfe das Raster und seine Werte
  print(prediction_raster)
  print(summary(values(prediction_raster)))  # Zeigt die Verteilung der Vorhersagewerte
  
  # Plotten des Vorhersage-Rasters
  plot(prediction_raster, main = "Vorhersage-Raster", col = terrain.colors(6))  # 6 Klassenfarben
  message("Vorhersage-Raster erfolgreich geplottet.")
}




# Beispiel für die Verwendung der Funktion
aoi_cube <- extract_values(bbox = aoi_target, crop_box = bbox_target_utm, assets = assets)
onnx_model_path <- "random_forest_model.onnx"

predictors <- c("B02", "B03", "B04", "B08")
predict_with_onnx_model(onnx_model_path, aoi_cube, predictors)