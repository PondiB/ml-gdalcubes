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

##############################################################################
# Extrahiere das CRS des Datacubes
crs_data <- extract_crs_datacube(aot_cube = aot_cube_multi)

# Transformiere die Trainingsdaten ins CRS des Datacubes
train_dat <- transform_training_data(train_dat = training_sites, aot_crs = crs_data)
extraction <- gdalcubes::extract_geom(cube = aot_cube_multi, sf = train_dat)
# Filtern der Daten für September 2021 mit 'subset'
extraction_sep <- subset(extraction, format(as.Date(time), "%Y-%m") == "2021-09")

# Anzeigen der gefilterten Daten
print("Datenpunkte vom September 2021:")
print(extraction_sep)

##############################################################################
data_preprocessing_multi <- function(aot_cube, train_dat) {
  print("Starting data extraction...")
  
  # Datenextraktion aus dem Würfel basierend auf den Trainingsdaten
  extraction <- extraction
  
  print("Extraction result:")
  print(extraction)
  
  unique_times <- unique(extraction$time)
  num_times <- length(unique_times)
  print(paste("Du hast", num_times, "Zeitpunkte:"))
  print(unique_times)
  
  if (nrow(extraction) == 0) {
    stop("No data extracted. Check if the bounding boxes of the training data and the data cube overlap.")
  }
  
  train_dat$PolyID <- seq_len(nrow(train_dat))
  
  # Merge der extrahierten Daten mit den Trainingsdaten
  extraction <- base::merge(extraction, train_dat, by.x = "FID", by.y = "PolyID")
  message("Extraction merged with training data ....")
  
  # Entferne die geometry-Spalte vor der Bereinigung
  train_data <- extraction[, !(names(extraction) %in% c("geometry"))]
  
  
  # Dynamisch die Bandnamen aus dem Cube extrahieren
  band_names <- get_cube_band_names(aot_cube)
  
  # **Zeitvariable als numerischen Prädiktor hinzufügen**
  train_data$time_numeric <- as.numeric(as.Date(train_data$time))
  
  print("Struktur von train_data:")
  str(train_data)
  
  predictor_columns <- c(band_names, "time_numeric")  
  
  # Überprüfen, wie viele fehlende Werte vorhanden sind
  na_counts <- colSums(is.na(train_data[, predictor_columns]))
  print("Anzahl fehlender Werte pro Prädiktor:")
  print(na_counts)
  
  # Entferne Zeilen mit fehlenden Werten in den Prädiktoren
  train_data <- train_data[stats::complete.cases(train_data[, predictor_columns]), ]
  
  
  print("Anzahl der Zeilen nach der Bereinigung:")
  print(nrow(train_data))
  
  message("Training data prepared (multiple time steps handled if applicable). . . .")
  return(train_data)
}

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




# if the data is not numerical, it will be changed
numeric <- function(data){
  if (!is.numeric(data)) {
    data <- as.factor(data)
  }
  print(data)
  return(data)
}

# get cube names
get_cube_band_names <- function(cube) {
  band_names <- names(cube)
  return(band_names)
}



ml_fit_class_random_forest <- function(aot_cube, training_set, target_column, num_trees = 100, max_variables = NULL, seed = NULL, multiple_time_steps = FALSE) {
  # Extrahiere das CRS des Datacubes
  crs_data <- extract_crs_datacube(aot_cube = aot_cube)
  
  # Transformiere die Trainingsdaten ins CRS des Datacubes
  train_dat <- transform_training_data(train_dat = training_set, aot_crs = crs_data)
  
  if (multiple_time_steps) {
    train_data <- data_preprocessing_multi(aot_cube = aot_cube, train_dat = train_dat)
    predictor_names <- c(get_cube_band_names(aot_cube), "time_numeric")
    
    x <- as.data.frame(lapply(train_data[, predictor_names], as.numeric))
  } else {
    train_data <- data_preprocessing_single(aot_cube = aot_cube, train_dat = train_dat)
    predictor_names <- get_cube_band_names(aot_cube)
    
    x <- train_data[, predictor_names]
  }
  
  message("Combining training data with cube data . . . .")
  
  y <- as.factor(train_data[[target_column]])
  
  # Setze den Seed für Reproduzierbarkeit
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # Bestimme mtry, falls nicht angegeben
  if (is.null(max_variables)) {
    max_variables <- floor(sqrt(ncol(x)))
  }
  
  # Trainiere das Random-Forest-Modell
  model <- randomForest::randomForest(
    x = x,
    y = y,
    ntree = num_trees,
    mtry = max_variables,
    importance = TRUE
  )
  
  print(model) # Zeigt das trainierte Modell an
  
  return(list(model = model, train_data = train_data))
}
#############################################################



# train svm and return the model
ml_fit_class_svm <- function(aot_cube, training_set, target_column, 
                             kernel = "radial", C = 1, sigma = 1, 
                             gamma = NULL, degree = 3, coef0 = 0, seed = NULL,
                             multiple_time_steps = FALSE) {
  
  # Extrahiere das CRS des Datacubes
  crs_data <- extract_crs_datacube(aot_cube = aot_cube)
  
  # Transformiere die Trainingsdaten ins CRS des Datacubes
  train_dat <- transform_training_data(train_dat = training_set, aot_crs = crs_data)
  
  if (multiple_time_steps) {
    train_data <- data_preprocessing_multi(aot_cube = aot_cube, train_dat = train_dat)
    predictor_names <- c(get_cube_band_names(aot_cube), "time_numeric")
    
    x <- as.data.frame(lapply(train_data[, predictor_names], as.numeric))
  } else {
    train_data <- data_preprocessing_single(aot_cube = aot_cube, train_dat = train_dat)
    predictor_names <- get_cube_band_names(aot_cube)
    
    x <- train_data[, predictor_names]
  }
  
  message("Combining training data with cube data . . . .")
  
  # Extrahiere die Zielvariable
  y <- as.factor(train_data[[target_column]])
  
  # Setze den Seed für Reproduzierbarkeit
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # Auswahl des Tuningsets basierend auf dem Kerneltyp
  if (kernel == "radial") {
    tune_grid <- base::expand.grid(C = C, sigma = sigma)
    method <- "svmRadial"
  } else if (kernel == "linear") {
    tune_grid <- base::expand.grid(C = C)
    method <- "svmLinear"
  } else if (kernel == "polynomial") {
    tune_grid <- base::expand.grid(C = C, degree = degree, scale = gamma, coef0 = coef0)
    method <- "svmPoly"
  } else {
    stop("Unsupported kernel type. Choose from 'radial', 'linear', or 'polynomial'.")
  }
  
  train_control <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 3)
  
  # Training des SVM-Modells
  svm_model <- caret::train(
    x = x, 
    y = y,
    method = method,
    tuneGrid = tune_grid,
    trControl = train_control,
    preProcess = c("center", "scale")
  )
  
  message("Model trained")  
  
  return(list(model = svm_model, train_data = train_data))
}


# train xgboost and return the model
ml_fit_class_xgboost <- function(aot_cube, training_set, target_column, 
                                 learning_rate = 0.15, max_depth = 5, min_child_weight = 1, 
                                 subsample = 0.8, colsample_bytree = 1, min_split_loss = 1, 
                                 nrounds = 100, seed = NULL, multiple_time_steps = FALSE) {
  
  # Extrahiere das CRS des Datacubes
  crs_data <- extract_crs_datacube(aot_cube = aot_cube)
  
  # Transformiere die Trainingsdaten ins CRS des Datacubes
  train_dat <- transform_training_data(train_dat = training_set, aot_crs = crs_data)
  
  if (multiple_time_steps) {
    train_data <- data_preprocessing_multi(aot_cube = aot_cube, train_dat = train_dat)
    predictor_names <- c(get_cube_band_names(aot_cube), "time_numeric")
    
    x <- as.data.frame(lapply(train_data[, predictor_names], as.numeric))
  } else {
    train_data <- data_preprocessing_single(aot_cube = aot_cube, train_dat = train_dat)
    predictor_names <- get_cube_band_names(aot_cube)
    
    x <- train_data[, predictor_names]
  }
  
  message("Combining training data with cube data . . . .")
  
  # Extrahiere die Zielvariable und konvertiere sie in numerische Werte
  y <- as.factor(train_data[[target_column]])
  
  # Setze den Seed für Reproduzierbarkeit
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # Setzt die Parameter für das XGBoost-Modell
  tune_grid <- expand.grid(
    nrounds = nrounds,
    max_depth = max_depth,
    eta = learning_rate,
    gamma = min_split_loss,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    subsample = subsample
  )
  
  # Training des XGBoost-Modells mit caret
  model <- caret::train(
    x = x, 
    y = y,
    method = "xgbTree",
    trControl = trainControl(method = "cv", number = 5, search = "grid"),
    tuneGrid = tune_grid
  )
  
  message("Model trained.")
  
  # Rückgabe des trainierten Modells und der Trainingsdaten
  return(list(model = model, train_data = train_data))
}


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


# Funktion zur Erstellung eines Data Cubes mit benutzerdefinierter Pixel-Anwendung
create_cube_with_custom_expr <- function(aoi_cube_multi, expr, names) {
  # Überprüfe, ob expr und names gleich lang sind
  if (length(expr) != length(names)) {
    stop("Die Länge von 'expr' und 'names' muss übereinstimmen.")
  }
  
  aoi_cube_custom <- gdalcubes::apply_pixel(
    x = aoi_cube_multi,
    expr = expr,
    names = names
  )
  
  return(aoi_cube_custom)
}





ml_predict_mean <- function(data_cube, model) {
  tryCatch({
    prediction <- predict(data_cube, model)
    message("Vorhersage für jeden Zeitschritt abgeschlossen.")
    
    # Konvertiere die Vorhersagen in einen terra Raster
    prediction_raster <- terra::rast(gdalcubes::write_tif(prediction))
    
    # Berechnung des Mittelwerts über alle Zeitschritte
    mean_raster <- terra::app(prediction_raster, fun = mean, na.rm = TRUE)
    message("Mittelwert pro Pixel über alle Zeitschritte berechnet.")
    
    # Berechne die minimalen und maximalen Werte explizit mit global()
    minmax_values <- terra::global(prediction_raster, fun = range, na.rm = TRUE)
    
    min_label <- min(minmax_values[, 1], na.rm = TRUE)
    max_label <- max(minmax_values[, 2], na.rm = TRUE)
    
    message(paste("Minimales Label:", min_label))
    message(paste("Maximales Label:", max_label))
    
    if (!is.finite(min_label) | !is.finite(max_label)) {
      stop("Keine gültigen minimalen oder maximalen Labels gefunden.")
    }
    
    # Dynamische Klassengrenzen erstellen
    step_size <- 1
    breakpoints <- seq(min_label - 0.5, max_label + 0.5, by = step_size)
    classification_matrix <- cbind(breakpoints[-length(breakpoints)], breakpoints[-1], min_label:max_label)
    
    # Klassifizierung der Mittelwerte
    classified_mean_raster <- terra::classify(mean_raster, classification_matrix)
    message("Mittelwert klassifiziert.")
    
    return(list(prediction = prediction_raster, mean = classified_mean_raster))
    
  },
  error = function(e) {
    message("Fehler bei der Klassifikation:")
    message(conditionMessage(e))
  })
}



plot_prediction_steps <- function(prediction_raster, mean_raster) {
  # Anzahl der Zeitschritte extrahieren
  n_steps <- terra::nlyr(prediction_raster)
  
  par(mfrow = c(1, n_steps + 1))
  
  for (i in 1:n_steps) {
    plot(prediction_raster[[i]], main = paste("Prediction Step", i))
  }
  
  plot(mean_raster, main = "Mean of All Steps (Classified)")
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


# call up functions
aot_cube<- extract_values(bbox = bbox, crop_box = aot_test, assets = assets )
aoi_cube <- extract_values(bbox = aoi_target, crop_box = bbox_target_utm, assets = assets)
aot_cube_multi<- extract_values_multi(bbox = bbox, crop_box = aot_test, assets = assets )
aoi_cube_multi <- extract_values_multi(bbox = aoi_target, crop_box = bbox_target_utm, assets = assets)

### random forest 
## random forest single
rf_result <- ml_fit_class_random_forest(aot_cube = aot_cube, training_set = training_sites, target_column = "Label", multiple_time_steps = FALSE)
random_forest_model <- rf_result$model
train_data_rf <- rf_result$train_data
select_band <- get_cube_band_names(aoi_cube)
r_ml <- ml_predict(data_cube = aoi_cube, model = random_forest_model)
prediction_raster_rf <- terra::rast(gdalcubes::write_tif(r_ml))
save_model_as_rds(model = random_forest_model, filepath = "random_forest_model.rds")
save_model_as_onnx(random_forest_model, "random_forest", "random_forest_model.onnx", select_band, train_data_rf)
plot(prediction_raster_rf)
### random forest multi
rf_result_multi <- ml_fit_class_random_forest(aot_cube = aot_cube_multi, training_set = training_sites, target_column = "Label", multiple_time_steps = TRUE)
rf_model <- rf_result_multi$model
rf_training <- rf_result_multi$train_data
expr <- c("B02", "B03", "B04", "B08", "t0 / 86400")
names <- c("B02", "B03", "B04", "B08", "time_numeric")
aoi_cube_with_time_numeric <- create_cube_with_custom_expr(aoi_cube_multi, expr, names)
result <- ml_predict_mean(data_cube = aoi_cube_with_time_numeric, model = rf_model)
prediction_raster_rf_multi <- result$prediction
mean_raster <- result$mean
plot_prediction_steps(prediction_raster_rf_multi, mean_raster)


### SVM 
### svm single
svm_classification <- ml_fit_class_svm(aot_cube = aot_cube, training_set = training_sites, target_column = "Label", multiple_time_steps = FALSE)
s_ml_model <- svm_classification$model
s_ml_train_data <- svm_classification$train_data
s_ml <- ml_predict(data = aoi_cube, model = s_ml_model)
prediction_raster_svm <- terra::rast(gdalcubes::write_tif(s_ml))
save_model_as_onnx(s_ml_model, "svm", "svm.onnx", select_band, s_ml_train_data)
save_model_as_rds(model = s_ml_model, filepath = "svm_model.rds")
plot(prediction_raster_svm)

### svm multi
svm_result_multi <- ml_fit_class_svm(aot_cube = aot_cube_multi, training_set = training_sites, target_column = "Label", multiple_time_steps = TRUE)
svm_model <- svm_result_multi$model
svm_train_data <- svm_result_multi$train_data
expr <- c("B02", "B03", "B04", "B08", "t0 / 86400")
names <- c("B02", "B03", "B04", "B08", "time_numeric")
aoi_cube_with_time_numeric <- create_cube_with_custom_expr(aoi_cube_multi, expr, names)
result <- ml_predict_mean(data_cube = aoi_cube_with_time_numeric, model = svm_model)
prediction_raster_rf_multi <- result$prediction
mean_raster <- result$mean
plot_prediction_steps(prediction_raster_rf_multi, mean_raster)


### xgboost
### xgboost single
xgboost_model <- ml_fit_class_xgboost(aot_cube = aot_cube, training_set = training_sites, target_column = "Label", multiple_time_steps = FALSE)
xgb_model <- xgboost_model$model
xgb_train_data <- xgboost_model$train_data

xgboost_prediction <- ml_predict(data = aoi_cube, model = xgb_model)
xgboost_prediction_raster <- terra::rast(gdalcubes::write_tif(xgboost_prediction))
save_model_as_rds(model = xgboost_model, filepath = "xgboost_model.rds")
xgboost::xgb.save(xgb_model$finalModel, "xgboost_native_model.bin")
save_model_as_onnx(xgb_model, "xgboost", "xgboost_model.onnx", select_band, xgb_train_data)
plot(xgboost_prediction_raster)

### xgboost multi
xgboost_result_multi <- ml_fit_class_xgboost(aot_cube = aot_cube_multi, training_set = training_sites, target_column = "Label", multiple_time_steps = TRUE)
xgboost_model <- xgboost_result_multi$model
xgboost_train_data <- xgboost_result_multi$train_data
expr <- c("B02", "B03", "B04", "B08", "t0 / 86400")
names <- c("B02", "B03", "B04", "B08", "time_numeric")
aoi_cube_with_time_numeric <- create_cube_with_custom_expr(aoi_cube_multi, expr, names)
result <- ml_predict_mean(data_cube = aoi_cube_with_time_numeric, model = xgboost_model)
prediction_raster_rf_multi <- result$prediction
mean_raster <- result$mean
plot_prediction_steps(prediction_raster_rf_multi, mean_raster)



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