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
library(torch)
library(abind)

####################################################################

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
  # Create the data cube with the defined view
  data_cube <- gdalcubes::raster_cube(s2_collection_training, cube) %>%
    gdalcubes::apply_pixel(c("B02", "B03", "B04", "B08", "(B08-B04)/(B08+B04)"),
                           names = c("B02", "B03", "B04", "B08", "NDVI")) %>%
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
    gdalcubes::apply_pixel(c("B02", "B03", "B04", "B08", "(B08-B04)/(B08+B04)"),
                           names = c("B02", "B03", "B04", "B08", "NDVI")) %>%
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



####################################################################
### ml_fit() 
ml_fit <- function(model, training_set, target_column) {
  if (!is.null(model$parameters$cnn_layer)) {
    message("Deep learning model recognized. Start DL calculation...")
    
    # Features extrahieren
    extracted_data <- extract_features(training_set, label_column = target_column)
    features_data <- extracted_data$band
    time_steps <- extracted_data$time_steps
    class_count <- extracted_data$label_count
  
    
    features <- extract_time_series_features(training_set, features_data, time_steps)
    print(features)
    
    labels <- as.numeric(as.factor(training_set[[target_column]]))
    x_train <- torch_tensor(features, dtype = torch_float())
    y_train <- torch_tensor(labels, dtype = torch_long())
    print(dim(x_train))
    
    # Deep-Learning-Modell erstellen
    dl_model <- model$create_model(
      input_data_columns = features_data,
      time_steps = time_steps,
      class_count = class_count
    )
    
    optimizer <- optim_adam(dl_model$parameters, lr = model$parameters$learning_rate, weight_decay = model$parameters$weight_decay)
    loss_fn <- nn_cross_entropy_loss()
    
    # Training Loop
    for (epoch in 1:model$parameters$epochs) {
      dl_model$train()
      optimizer$zero_grad()
      predictions <- dl_model(x_train)
      loss <- loss_fn(predictions, y_train)
      loss$backward()
      optimizer$step()
      print(sprintf("Epoch: %d, Loss: %.4f", epoch, loss$item()))
    }
    
    dl_model$eval()
    with_no_grad({
      predictions <- dl_model(x_train)
      predicted_classes <- torch_argmax(predictions, dim = 2)
      accuracy <- mean(as.numeric(predicted_classes == y_train))
      print(sprintf("Accuracy: %.2f%%", accuracy * 100))
    })
    
    confusion_matrix <- table(Predicted = as.numeric(predicted_classes), Actual = as.numeric(y_train))
    print(confusion_matrix)
    
    return(dl_model)
  }
  
  message("Machine learning model recognized. Start ML calculation...")
  
  y <- training_set[[target_column]]
  if (!is.numeric(y)) {
    y <- as.factor(y)
    message("Classification is carried out...")
  } else {
    y <- as.numeric(y)
    message("Regression is carried out...")
  }
  
  predictor_names <- identify_predictors(training_set)
  
  
  if (length(predictor_names) == 0) {
    stop("No valid predictors detected. Please check.")
  }
  message("Automatically recognized predictors: ", paste(predictor_names, collapse = ", "))
  
  x <- as.data.frame(lapply(training_set[, predictor_names, drop = FALSE], as.numeric))
  
  if (ncol(x) == 0) {
    stop("No predictors detected.")
  }
  
  if (model$method == "rf") {
    # Random Forest spezifische Einstellungen
    if (is.null(model$tuneGrid) || is.na(model$tuneGrid$mtry)) {
      max_variables <- max(1, floor(sqrt(ncol(x))))  
      model$tuneGrid <- expand.grid(mtry = max_variables)
      message("tuneGrid was automatically set to mtry = ", max_variables)
    } else if (model$tuneGrid$mtry < 1 || model$tuneGrid$mtry > ncol(x)) {
      warning("Invalid `mtry` value in tuneGrid. Set `mtry` to 1.")
      model$tuneGrid <- expand.grid(mtry = 1)
    }
  } else if (model$method %in% c("svmRadial", "svmLinear", "svmPoly")) {
    if (is.null(model$tuneGrid)) {
      stop("SVM models require a defined tuneGrid")
    }
    model$preProcess <- c("center", "scale")
  } else if (model$method == "xgbTree") {
    if (is.null(model$tuneGrid)) {
      stop("XGBoost model require a defined tuneGrid")
    }
    if (is.null(model$trControl)) {
      model$trControl <- caret::trainControl(method = "cv", number = 5, search = "grid")
    }
  } else {
    stop("Undetected method! You can only work with: 'rf', 'svmRadial', 'svmLinear', 'svmPoly', 'xgbTree'.")
  }
  
  model <- caret::train(
    x = x,
    y = y,
    method = model$method,
    tuneGrid = model$tuneGrid,
    trControl = model$trControl,
    ntree = if (model$method == "rf") model$ntree else NULL,
    preProcess = if (model$method %in% c("svmRadial", "svmLinear", "svmPoly")) model$preProcess else NULL
  )
  
  # Accuracy oder RMSE ausgeben
  if (!is.numeric(y)) {
    if ("Accuracy" %in% colnames(model$results)) {
      accuracy <- max(model$results$Accuracy, na.rm = TRUE)
      message("Accuracy: ", round(accuracy * 100, 2), "%")
    }
  }
  
  return(model)
}


ml_predict <- function(cube, model) {
  is_dl_model <- inherits(model, "nn_module")
  band_info <- gdalcubes::bands(cube)
  band_names <- band_info$name
  cube_dimensions <- gdalcubes::dimensions(cube)
  time_count <- cube_dimensions$t$count  
  multi_timesteps <- time_count > 1
   
  
  if (is_dl_model) {
    message("Deep Learning Model detected")
    if (multi_timesteps) {
      message("More time steps detected.")
      time_steps <- time_steps_query(cube = cube)
      features <- prepare_and_combine_multitemporal_cube(cube, time_steps = time_steps, model = model)
    } else {
        message("One time step detected")
        cube_data <- extract_pixel_values(cube, band_names)
        if (is.null(cube_data) || nrow(cube_data) == 0) {
          stop("The extracted data is empty. Check the cube and the tapes.")
        }
        features <- array(
          data = as.matrix(cube_data),
          dim = c(nrow(cube_data), ncol(cube_data), 1) 
        )
      
    }

    aoi_tensor <- torch_tensor(features, dtype = torch_float())
  
    model$eval()
    with_no_grad({
      predictions <- model(aoi_tensor)
      cat("Shape of predictions:", dim(predictions), "\n")
      predicted_classes <- torch_argmax(predictions, dim = 2)
    })

    predicted_classes_vector <- as.numeric(as_array(predicted_classes))
  
    prediction_deep_learning <- create_raster_from_cube(
      cube = cube, 
      predictions = predicted_classes_vector 
    )

    return(prediction_deep_learning)


  } else {
    message("Machine Learning detected")
    # Nutze ML Predict-Logik
    if (multi_timesteps) {
      message("More time steps detected")
      time_steps <- time_steps_query(cube = cube)
      combined_data_cube <- prepare_and_combine_multitemporal_cube(cube, time_steps, model)
      combined_data <- combined_data_cube$combined_data
      new_names <- combined_data_cube$band_names
      predictions <- clean_and_predict(combined_data, new_names, model)
      predicted_classes_vector <- as.numeric(predictions)
      prediction_raster <- create_raster_from_cube(cube, predicted_classes_vector)
      return(prediction_raster)
    } else {
      message("One Time Step detected")
      prediction_raster <- mlm_predict_single(cube, model)      
      return(prediction_raster)
    }
  }
}




###################################################################
## help functions


#Extracts the coordinate system (CRS) of the data cube
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

get_train_data <- function(file_path, default_crs = FALSE) {
  if (grepl("\\.geojson$", file_path)) {
    message("Load GeoJSON-Data...")
    training_data <- sf::read_sf(file_path)
    message("GeoJSON-Daten loaded")
    
  } else if (grepl("\\.csv$", file_path)) {
    message("Load CSV-Data...")
    csv_data <- read_csv(file_path)
    
    convert_to_sf <- function(geom_str) {
      coords <- fromJSON(geom_str)
      if (!all(coords[1, ] == coords[nrow(coords), ])) {
        coords <- rbind(coords, coords[1, ])
      }
      polygon <- st_polygon(list(coords))
      return(polygon)
    }
    
    if (!inherits(csv_data$geometry, "sfc")) {
      if (grepl("^\\[\\[", csv_data$geometry[1])) {
        message("Convert the geometry column from string to sf...")
        csv_data$geometry <- st_sfc(lapply(csv_data$geometry, convert_to_sf))
        message("Geometry converted as sf objects.")
      } else {
        csv_data$geometry <- st_as_sfc(csv_data$geometry, GeoJSON = TRUE)
        message("Geometry converted as GeoJSON.")
      }
    }
    
    training_data <- st_as_sf(csv_data)
    message("CSV data saved as sf object.")
    
    if (is.null(sf::st_crs(training_data)) || is.na(sf::st_crs(training_data))) {
      if (!isFALSE(default_crs)) {
        message(paste("Set the CRS specified by the user (EPSG:", default_crs, ")...", sep = ""))
        sf::st_crs(training_data) <- default_crs
      } else {
        stop("CRS is missing in the data and no default CRS was specified.")
      }
    }
    
    if (!"fid" %in% colnames(training_data)) {
      message("Create a unique `fid` column...")
      training_data$fid <- seq_len(nrow(training_data))  
    } else {
      training_data$fid <- as.integer(training_data$fid)  
    }
    
  } else {
    stop("File format not supported. Please use CSV or GeoJSON file.")
  }
  
  training_data <- training_data %>%
    select(-geometry, everything(), geometry)
  
  return(training_data)
}


data_preprocessing_single <- function(aot_cube, train_dat){
  print("Starting data extraction...")
  
  if (!"fid" %in% colnames(train_dat)) {
    stop("The column `fid` is missing in the training data.")
  }
  train_dat$fid <- as.integer(train_dat$fid)  
  
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
  if (!"fid" %in% colnames(train_dat)) {
    stop("“The column `fid` is missing in the training data")
  }
  train_dat$fid <- as.integer(train_dat$fid)  # `fid` als Integer sicherstellen
  
  extraction <- gdalcubes::extract_geom(
    cube = aot_cube,
    sf = train_dat
  )  
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

convert_to_wide_format <- function(train_data, band_names) {
  time_steps <- unique(train_data$time)
  n_steps <- length(time_steps)
  
  data_frames <- list()
  
  has_ndvi <- "NDVI" %in% colnames(train_data)
  
  for (i in seq_along(time_steps)) {
    data_time <- train_data %>%
      filter(time == time_steps[i]) %>%
      dplyr::select(-geometry) %>%
      dplyr::rename_with(
        ~ paste0(., "_T", i),
        all_of(c(band_names, if (has_ndvi) "NDVI"))
      ) %>%
      dplyr::select(FID, pixel_id, starts_with("B"), if (has_ndvi) starts_with("NDVI"), -time, -time_numeric)
    
    data_time$pixel_id <- seq_len(nrow(data_time))
    
    data_frames[[i]] <- data_time
    print(paste("Rename data for time step", i))
    print(data_time)
  }
  
  wide_data <- Reduce(function(x, y) inner_join(x, y, by = c("FID", "pixel_id")), data_frames)
  
  complete_data <- wide_data[complete.cases(wide_data), ]
  
  message("Data in wide format after merging the time steps:")
  print(complete_data)
  
  return(complete_data)
}



get_cube_band_names <- function(cube) {
  band_names <- names(cube)
  return(band_names)
}

identify_predictors <- function(training_set, pattern = "^(B\\d+|(?i)NDVI(_T\\d+)?)$") {
  predictor_names <- colnames(training_set)
  predictor_names <- predictor_names[
    grepl(pattern, predictor_names) & 
      sapply(training_set[, predictor_names, drop = FALSE], is.numeric)
  ]
  if (length(predictor_names) == 0) {
    stop("No valid predictors detected. Please check.")
  }
  return(predictor_names)
}

extract_time_series_features <- function(training_set, features_data, time_steps) {
  if (time_steps > 1) {
    features <- array(
      data = as.matrix(training_set[, grep("_T\\d+$", colnames(training_set))]),
      dim = c(nrow(training_set), length(features_data), time_steps)
    )
  } else {
    features <- array(
      data = as.matrix(training_set[, features_data]),
      dim = c(nrow(training_set), length(features_data), time_steps)
    )
  }
  return(features)
}

preprocess_training_set <- function(
    training_set, 
    aot_cube, 
    target_column, 
    multiple_time_steps = FALSE, 
    is_classification = NULL, 
    srs_train_data = FALSE
) {
  training_set <- get_train_data(training_set, srs_train_data) 
  
  crs_data <- extract_crs_datacube(aot_cube)
  train_dat <- transform_training_data(train_dat = training_set, aot_crs = crs_data)
  
  y <- train_dat %>%
    st_set_geometry(NULL) %>%
    dplyr::pull(!!rlang::sym(target_column))
  
  if (is.null(is_classification)) {
    is_classification <- is.factor(y) || is.character(y)
    message(if (is_classification) "Classification detected." else "Regression detected.")
  } else {
    message(if (is_classification) "Explicit classification set." else "Explicit regression set")
  }
  
  if (is_classification && !is.factor(y)) {
    message("Numerical target variable is converted into a factor.")
    train_dat[[target_column]] <- as.factor(train_dat[[target_column]])
  }
  
  if (multiple_time_steps) {
    train_data <- data_preprocessing_multiple(aot_cube = aot_cube, train_dat = train_dat)
    band_names <- get_cube_band_names(aot_cube)
    train_data_wide <- convert_to_wide_format(train_data = train_data, band_names = band_names)
    
    colnames(train_data_wide)[colnames(train_data_wide) == "FID"] <- "fid"
    train_data <- merge(
      train_data_wide, 
      train_dat %>%
        st_set_geometry(NULL) %>%
        dplyr::select(fid, !!rlang::sym(target_column)),
      by = "fid", 
      all.x = TRUE
    )
  } else {
    train_data <- data_preprocessing_single(aot_cube = aot_cube, train_dat = train_dat)
    
    train_data_no_geom <- train_data %>%
      dplyr::select(-geometry)  # Geometrie-Spalte entfernen
    
    train_data_no_geom <- train_data_no_geom[complete.cases(train_data_no_geom), ]
    
    geometry_data <- train_dat %>%
      dplyr::select(fid, geometry)
    
    train_data <- merge(train_data_no_geom, geometry_data, by = "fid", all.x = TRUE)
  }
  
  return(train_data)
}


extract_features <- function(data, pattern = "_T\\d+$", bands = NULL, label_column = NULL) {
  feature_names <- colnames(data)
  
  if (!is.null(bands)) {
    time_step_columns <- grep(pattern, feature_names, value = TRUE)
    unique_steps <- unique(gsub(".*_T", "", time_step_columns))
    if (length(unique_steps) == 0) {
      unique_steps = 1
    }
    
    label_count <- if (!is.null(label_column)) {
      length(unique(data[[label_column]]))  
    } else {
      NA  
    }
    
    return(list(
      band = bands,
      time_steps = length(unique_steps),  
      label_count = label_count  
    ))
  }
  
  is_band <- function(name) {
    grepl("^(B\\d{2}|NDVI)$", name, ignore.case = TRUE)
  }
  
  if (any(grepl(pattern, feature_names))) {
    time_step_columns <- grep(pattern, feature_names, value = TRUE)
    
    unique_steps <- unique(gsub(".*_T", "", time_step_columns))
    
    relevant_features <- unique(gsub(pattern, "", time_step_columns))
    
    relevant_features <- relevant_features[sapply(relevant_features, is_band)]
    
    label_count <- if (!is.null(label_column)) {
      length(unique(data[[label_column]]))  
    } else {
      NA  
    }
    
    return(list(
      band = relevant_features,
      time_steps = length(unique_steps),
      label_count = label_count  
    ))
  } else {
    relevant_features <- feature_names[sapply(feature_names, is_band)]
    
    label_count <- if (!is.null(label_column)) {
      length(unique(data[[label_column]]))  
    } else {
      NA  
    }
    
    return(list(
      band = relevant_features,
      time_steps = 1,
      label_count = label_count  
    ))
  }
}


##################################################################


mlm_predict_single <- function(data_cube, model, dimension = NULL) {
  tryCatch({
    prediction <- predict(aoi_cube, model)
    print(prediction)
    message("Prediction calculated ....")
    message(gdalcubes::as_json(prediction))
    prediction <- terra::rast(gdalcubes::write_tif(prediction))
    return(prediction)
  },
  error = function(e){
    message("Error in classification")
    message(conditionMessage(e))
  })
}


time_steps_query <- function(cube){
  time_steps <- gdalcubes::dimension_values(aoi_cube_multi)
  time_steps <- time_steps$t
  return(time_steps)
}


clean_and_predict <- function(data_frame, new_names, model) {
  cleaned_data <- data_frame[, colSums(is.na(data_frame)) == 0]
  print(cleaned_data)

  if (length(new_names) == ncol(cleaned_data)) {
    names(cleaned_data) <- new_names
  } else {
    stop("The number of new column names does not match the number of columns in the cleansed DataFrame.")
  }
  
  predictions <- predict(model, cleaned_data)
  
  return(predictions)
}




create_raster_from_cube <- function(cube, predictions) {
  srs <- gdalcubes::srs(cube)
  dims <- gdalcubes::dimensions(cube)
  
  nx <- dims$x$count
  ny <- dims$y$count
  
  rast_template <- terra::rast(ncols = nx, nrows = ny)
  
  terra::ext(rast_template) <- terra::ext(dims$x$low, dims$x$high, dims$y$low, dims$y$high)
  
  if (!is.null(srs)) {
    terra::crs(rast_template) <- srs
  } else {
    stop("Please enter a valid 'srs' parameter")
  }
  
  prediction_raster <- rast_template
  print(predictions)
  values(prediction_raster) <- as.vector(predictions)
  
  return(prediction_raster)
}

prepare_and_combine_multitemporal_cube <- function(cube, time_steps, model) {
  is_dl_model <- inherits(model, "nn_module")
  
  band_info <- gdalcubes::bands(cube)
  band_names <- band_info$name
  print(band_names)
  
  data_frames <- vector("list", length(time_steps))
  all_band_names <- c()  
  
  for (i in seq_along(time_steps)) {
    data_cube <- gdalcubes::select_bands(cube, band_names)
    data_cube <- gdalcubes::slice_time(data_cube, time_steps[i])
    
    if (is_dl_model) {
      data_frame <- as.data.frame(gdalcubes::reduce_time(data_cube, FUN = function(data) { return(data) }))
      
      if (nrow(data_frame) == 0) {
        stop(paste("Zeitschritt", time_steps[i], "ist leer."))
      }
      
      cleaned_data <- data_frame[, colSums(is.na(data_frame)) == 0]
      cleaned_data <- cleaned_data[, colSums(cleaned_data != 0) > 0]
      
      cat("Zeitschritt:", i, "\n")
      print(summary(cleaned_data))
      
      data_frames[[i]] <- as.matrix(cleaned_data)
    } else {
      renamed_bands <- paste0(band_names, "_T", i)
      all_band_names <- c(all_band_names, renamed_bands)  
      
      rename_map <- setNames(renamed_bands, band_names)
      data_cube <- do.call(gdalcubes::rename_bands, c(list(data_cube), rename_map))
      
      data_frame <- as.data.frame(gdalcubes::reduce_time(data_cube, FUN = function(data) { return(data) }))
      
      data_frames[[i]] <- data_frame
    }
  }
  
  if (is_dl_model) {
    for (i in seq_along(data_frames)) {
      if (!identical(dim(data_frames[[1]]), dim(data_frames[[i]]))) {
        stop(paste("Time steps", i, "has different dimensions:", dim(data_frames[[i]])))
      }
    }
    combined_data <- abind::abind(data_frames, along = 3)
    return(combined_data)
  } else {
    combined_data <- do.call(cbind, data_frames)
    return(list(combined_data = combined_data, band_names = all_band_names))
  }
}



extract_pixel_values <- function(cube, bands) {
  
  data_frames <- list()
  

    cube <- gdalcubes::select_bands(cube, bands)
    data_frame <- as.data.frame(gdalcubes::reduce_time(cube, FUN = function(data) { return(data) }))
    cleaned_data <- data_frame[, colSums(is.na(data_frame)) == 0]
    
    if (ncol(cleaned_data) == length(bands)) {
      colnames(cleaned_data) <- bands  
    } else {
      warning("The number of columns in the DataFrame does not match the number of selected bands.")
    }
    data_frames <- cleaned_data

    
  return(data_frames)
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


check_train_data <- function(trainings_data, predictors){
  
  missing_predictors <- setdiff(predictors, names(trainings_data))
  if (length(missing_predictors) > 0) {
    stop(paste("Fehlende Prädiktoren:", paste(missing_predictors, collapse = ", ")))
  }
  train_data_clean <- trainings_data
  train_data_clean[predictors] <- lapply(train_data_clean[predictors], function(x) as.numeric(as.character(x)))
  train_data_clean <- na.omit(train_data_clean) 
  return(train_data_clean)
}

conv_numeric <- function(train_data_clean){
  train_data_clean$Label <- as.numeric(as.factor(train_data_clean$Label)) - 1  
  return(train_data_clean)
}


# flexibler zugriff über target_column. Der Benutzer hat die Möglichkeit zu entscheiden, ob das Model eine 
# Regression oder eine Classification ist. Dabei wandelt die Funktion es bei einer Classification mit bsp. target_column = Label
# in  Integer Werte, hingegen bei der Regression mit bsp. target_column = temperature, bleiben die Werte numerisch. 
save_model_as_onnx <- function(model, model_type, filepath, predictors, train_data, target_column) {
  library(reticulate)
  
  # Importiere notwendige Python-Module
  onnxmltools <- import("onnxmltools")
  skl2onnx <- import("skl2onnx")
  xgboost <- import("xgboost")
  sklearn <- import("sklearn.ensemble")
  sklearn_svm <- import("sklearn.svm")  
  onnx <- import("onnx")
  np <- import("numpy")
  
  print(train_data)
  train_data_clean <- check_train_data(train_data, predictors)
  print(names(train_data_clean))
  
  if (is.factor(train_data_clean[[target_column]]) || is.character(train_data_clean[[target_column]])) {
    train_data_clean[[target_column]] <- as.integer(as.factor(train_data_clean[[target_column]])) - 1
  }
  
  print(class(train_data_clean))
  print(predictors)
  
  x_train <- np$array(as.matrix(train_data_clean[, predictors]))
  y_train <- np$array(as.numeric(train_data_clean[[target_column]]))  # Dynamisch basierend auf `target_column`
  
  FloatTensorType <- reticulate::import("skl2onnx.common.data_types")$FloatTensorType
  initial_type <- list(list("float_input", FloatTensorType(list(NULL, length(predictors)))))
  
  if (model_type == "xgboost") {
    xgb_model_py <- xgboost$XGBClassifier()
    xgb_model_py$load_model("xgboost_native_model.bin")
    
    onnx_model <- onnxmltools$convert_xgboost(xgb_model_py, initial_types = initial_type)
    
  } else if (model_type == "random_forest") {
    rf_model_py <- sklearn$RandomForestClassifier(n_estimators = as.integer(100))
    rf_model_py$fit(x_train, y_train)  
    onnx_model <- onnxmltools$convert_sklearn(rf_model_py, initial_types = initial_type)
    
  } else if (model_type == "svm") {
    svm_model_py <- sklearn_svm$SVC(kernel = "rbf", C = 1.0)
    svm_model_py$fit(x_train, y_train)  
    onnx_model <- skl2onnx$convert_sklearn(svm_model_py, initial_types = initial_type)
  }
  
  onnx$save_model(onnx_model, filepath)
  message(paste("Modell erfolgreich als ONNX gespeichert unter:", filepath))
}

##############################################################################
#Model cover
#rf
mlm_class_random_forest <- function(
    num_trees = 100,
    min_samples_split = 2,
    min_samples_leaf = 1,
    max_features = "sqrt",
    random_state = NULL
) {
  
  model_params <- list(
    method = "rf",
    tuneGrid = expand.grid(mtry = if (max_features == "sqrt") NA else max_features),
    trControl = caret::trainControl(
      method = "cv",  
      number = 5      
    ), 
    ntree = num_trees,
    seed = random_state,
    min_samples_split = min_samples_split,
    min_samples_leaf = min_samples_leaf
  )
  
  
  
  return(model_params)
}




mlm_regr_random_forest <- function(
    num_trees = 100,
    min_samples_split = 2,
    min_samples_leaf = 1,
    max_features = "sqrt",
    random_state = NULL
) {
  
  model_params <- list(
    method = "rf",
    tuneGrid = expand.grid(mtry = if (max_features == "sqrt") NA else max_features),
    trControl = caret::trainControl(
      method = "cv",  
      number = 5      
    ), 
    ntree = num_trees,
    seed = random_state,
    min_samples_split = min_samples_split,
    min_samples_leaf = min_samples_leaf
  )
  
  
  
  return(model_params)
}


mlm_random_forest <- function(num_trees,
                             min_samples_split,
                             min_samples_leaf,
                             max_features,
                             random_state,
                             classification = TRUE) {
  if (classification) {
    return(mlm_class_random_forest(num_trees,
                                  min_samples_split,
                                  min_samples_leaf,
                                  max_features,
                                  random_state))
  } else {
    return(mlm_regr_random_forest(num_trees,
                                 min_samples_split,
                                 min_samples_leaf,
                                 max_features,
                                 random_state))
  }
}

#svm
mlm_class_svm <- function(
    kernel = "radial", 
    C = 1, 
    sigma = NULL, 
    gamma = NULL, 
    degree = 3, 
    coef0 = 0, 
    random_state = NULL
) {
  if (kernel == "radial") {
    tuneGrid <- expand.grid(C = C, sigma = sigma)
    method <- "svmRadial"
  } else if (kernel == "linear") {
    tuneGrid <- expand.grid(C = C)
    method <- "svmLinear"
  } else if (kernel == "polynomial") {
    tuneGrid <- expand.grid(C = C, degree = degree, scale = gamma, coef0 = coef0)
    method <- "svmPoly"
  } else {
    stop("Unsupported kernel type. Choose from 'radial', 'linear', or 'polynomial'.")
  }
  
  model_params <- list(
    method = method,
    tuneGrid = tuneGrid,
    trControl = caret::trainControl(
      method = "cv",  
      number = 5      
    ),
    seed = random_state
  )
  
  return(model_params)
}

mlm_regr_svm <- function(
    kernel = "radial", 
    C = 1, 
    sigma = NULL, 
    gamma = NULL, 
    degree = 3, 
    coef0 = 0, 
    random_state = NULL
) {
  if (kernel == "radial") {
    tuneGrid <- expand.grid(C = C, sigma = sigma)
    method <- "svmRadial"
  } else if (kernel == "linear") {
    tuneGrid <- expand.grid(C = C)
    method <- "svmLinear"
  } else if (kernel == "polynomial") {
    tuneGrid <- expand.grid(C = C, degree = degree, scale = gamma, coef0 = coef0)
    method <- "svmPoly"
  } else {
    stop("Unsupported kernel type. Choose from 'radial', 'linear', or 'polynomial'.")
  }
  
  model_params <- list(
    method = method,
    tuneGrid = tuneGrid,
    trControl = caret::trainControl(
      method = "cv",  
      number = 5      
    ),
    seed = random_state
  )
  
  return(model_params)
}

mlm_svm <- function(kernel,
                   C,
                   sigma,
                   gamma,
                   degree,
                   coef0,
                   random_state,
                   classification = TRUE) {
  if (classification) {
    return(mlm_class_svm(
      kernel = kernel,
      C = C,
      sigma = sigma,
      gamma = gamma,
      degree = degree,
      coef0 = coef0,
      random_state = random_state
    ))
  } else {
    return(mlm_regr_svm(
      kernel = kernel,
      C = C,
      sigma = sigma,
      gamma = gamma,
      degree = degree,
      coef0 = coef0,
      random_state = random_state
    ))
  }
}

#xgb
mlm_class_xgboost <- function(
    learning_rate = 0.1,
    max_depth = 6,
    min_child_weight = 1,
    subsample = 0.8,
    colsample_bytree = 1,
    gamma = 0,
    nrounds = 100,
    random_state = NULL
) {
  model_params <- list(
    method = "xgbTree",
    tuneGrid = expand.grid(
      nrounds = nrounds,
      max_depth = max_depth,
      eta = learning_rate,
      gamma = gamma,
      colsample_bytree = colsample_bytree,
      min_child_weight = min_child_weight,
      subsample = subsample
    ),
    trControl = caret::trainControl(
      method = "cv",  
      number = 5,     
      search = "grid"
    ),
    random_state = random_state
  )
  
  return(model_params)
}

mlm_regr_xgboost <- function(
    learning_rate = 0.1,
    max_depth = 6,
    min_child_weight = 1,
    subsample = 0.8,
    colsample_bytree = 1,
    gamma = 0,
    nrounds = 100,
    random_state = NULL
) {
  model_params <- list(
    method = "xgbTree",
    tuneGrid = expand.grid(
      nrounds = nrounds,
      max_depth = max_depth,
      eta = learning_rate,
      gamma = gamma,
      colsample_bytree = colsample_bytree,
      min_child_weight = min_child_weight,
      subsample = subsample
    ),
    trControl = caret::trainControl(
      method = "cv",  
      number = 5,     
      search = "grid"
    ),
    random_state = random_state
  )
  
  return(model_params)
}

mlm_xgboost <- function(learning_rate,
                       max_depth,
                       min_child_weight,
                       subsample,
                       colsample_bytree,
                       gamma,
                       nrounds,
                       random_state,
                       classification = TRUE) {
  if (classification) {
    return(mlm_class_xgboost(
      learning_rate = learning_rate,
      max_depth = max_depth,
      min_child_weight = min_child_weight,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      gamma = gamma,
      nrounds = nrounds,
      random_state = random_state
    ))
  } else {
    return(mlm_regr_xgboost(
      learning_rate = learning_rate,
      max_depth = max_depth,
      min_child_weight = min_child_weight,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      gamma = gamma,
      nrounds = nrounds,
      random_state = random_state
    ))
  }
}



mlm_class_tempcnn <- function(cnn_layer = cnn_layer,
                             cnn_kernels = cnn_kernels,
                             cnn_dropout_rates = cnn_dropout_rates,
                             dense_layer_nodes = dense_layer_nodes,
                             dense_layer_dropout_rate = dense_layer_dropout_rate,
                             optimizer = optimizer,
                             learning_rate = learning_rate,
                             epsilon = epsilon,
                             weight_decay = weight_decay,
                             lr_decay_epochs = lr_decay_epochs,
                             lr_decay_rate = lr_decay_rate,
                             epochs = epochs,
                             batch_size = batch_size,
                             random_state = NULL) {
  
  model_parameter <- list(
    cnn_layer = cnn_layer,
    cnn_kernels = cnn_kernels,
    cnn_dropout_rates = cnn_dropout_rates,
    dense_layer_nodes = dense_layer_nodes,
    dense_layer_dropout_rate = dense_layer_dropout_rate,
    optimizer = optimizer,
    learning_rate = learning_rate,
    epsilon = epsilon,
    weight_decay = weight_decay,
    lr_decay_epochs = lr_decay_epochs,
    lr_decay_rate = lr_decay_rate,
    epochs = epochs,
    batch_size = batch_size,
    random_state = random_state
  )
  
  return(list(
    parameters = model_parameter,
    create_model = function(input_data_columns, time_steps, class_count) {
      dl_temp_cnn_dynamic <- nn_module(
        initialize = function(settings, input_data_columns, time_steps, class_count) {
          self$conv_layers <- nn_module_list()
          self$input_data_columns <- input_data_columns  
          self$original_time_steps <- time_steps         
          
          reduced_time_steps <- time_steps
          for (i in seq_along(settings$cnn_layer)) {
            input_channels <- ifelse(
              i == 1,
              length(self$input_data_columns),
              settings$cnn_layer[[i - 1]]
            )
            
            output_channels <- settings$cnn_layer[[i]]
            kernel_size <- settings$cnn_kernels[[i]]
            dropout_rate <- settings$cnn_dropout_rates[[i]]
            
            self$conv_layers$append(
              nn_sequential(
                nn_conv1d(
                  in_channels = input_channels,
                  out_channels = output_channels,
                  kernel_size = kernel_size,
                  stride = 1,
                  padding = kernel_size %/% 2
                ),
                nn_batch_norm1d(output_channels),
                nn_relu(),
                nn_dropout(p = dropout_rate)
              )
            )
            
            reduced_time_steps <- (reduced_time_steps + 2 * (kernel_size %/% 2) - kernel_size) + 1
          }
          
          self$time_steps <- reduced_time_steps
          self$flatten <- nn_flatten()
          self$dense <- nn_sequential(
            nn_linear(settings$cnn_layer[[length(settings$cnn_layer)]] * self$time_steps, settings$dense_layer_nodes),
            nn_relu(),
            nn_dropout(p = settings$dense_layer_dropout_rate),
            nn_linear(settings$dense_layer_nodes, class_count)
          )
        },
        
        forward = function(x) {
          for (i in seq_along(self$conv_layers)) {
            x <- self$conv_layers[[i]](x)
          }
          x <- self$flatten(x)
          x <- self$dense(x)
          return(x)
        }
      )
      # Erstelle und initialisiere das Modell
      return(dl_temp_cnn_dynamic(
        settings = model_parameter,
        input_data_columns = input_data_columns,
        time_steps = time_steps,
        class_count = class_count
      ))
    }
  ))
}


### filling model cases

### random forest
mlm_rf_regr <- mlm_regr_random_forest(
  num_trees = 200,
  max_features = "sqrt",
  random_state = 42,
  min_samples_split = 2, 
  min_samples_leaf = 1)



mlm_rf_class <- mlm_random_forest(
  num_trees = 400,
  max_features = "sqrt",
  random_state = 42,
  min_samples_split = 2, 
  min_samples_leaf = 1,
  classification = TRUE
)


### svm 

mlm_svm_class <- mlm_svm(
  kernel = "linear",
  C = 1,
  sigma = 0.1,
  random_state = 42,
  classification = TRUE
)


mlm_svm_regr <- mlm_svm(
  kernel = "linear",
  C = 1,
  sigma = 0.1,
  random_state = 42,
  classification = FALSE
)

### xgb

mlm_xgb_class <- mlm_xgboost(
  learning_rate = 0.15, 
  max_depth = 5, 
  min_child_weight = 1, 
  subsample = 0.8, 
  colsample_bytree = 1, 
  gamma = 1, 
  nrounds = 100,
  classification = TRUE, 
  random_state = NULL
)

mlm_xgb_regr <- mlm_xgboost(
  learning_rate = 0.15, 
  max_depth = 5, 
  min_child_weight = 1, 
  subsample = 0.8, 
  colsample_bytree = 1, 
  gamma = 1, 
  nrounds = 100,
  classification = FALSE, 
  random_state = NULL
)

### dl 

mlm_tempcnn <- mlm_class_tempcnn(
  cnn_layer = list(64, 64, 64),
  cnn_kernels = list(5, 5, 5),
  cnn_dropout_rates = list(0.2, 0.2, 0.2),
  dense_layer_nodes = 256,
  dense_layer_dropout_rate = 0.5,
  optimizer = "adam",
  learning_rate = 0.0005,
  epsilon = 0.00000001,
  weight_decay = 0.000001,
  lr_decay_epochs = 1,
  lr_decay_rate = 0.95,
  epochs = 200,
  batch_size = 64,
  random_state = 42
)



####### call functions #######################################################
# We need this to fill up the cubes. 
training_sites <- sf::read_sf("./train_data/train_dat.geojson")
assets <- c("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "SCL")

bbox <- sf::st_bbox(training_sites)
# Define the bounding box for the training area
aot_test <- sf::st_bbox(c(xmin = 388831.6, ymin = 5698900.1, xmax = 398063.3, ymax = 5718133.7), crs = sf::st_crs(25832))

aoi_target <- sf::st_bbox(
  c(
    xmin = 7.601035, ymin = 51.887296, 
    xmax = 7.668978, ymax = 51.916020
  ), 
  crs = sf::st_crs(4326)
)

# Conversion to an sf object
bbox_target_sf <- sf::st_as_sfc(aoi_target)

# Transform the coordinates to EPSG:25832
bbox_target_transformed <- sf::st_transform(bbox_target_sf, crs = sf::st_crs(25832))

# Extract the transformed coordinates
bbox_target_utm <- sf::st_bbox(bbox_target_transformed)



##### cube initialization

### one time step
aot_cube<- extract_values(bbox = bbox, crop_box = aot_test, assets = assets )
aoi_cube <- extract_values(bbox = aoi_target, crop_box = bbox_target_utm, assets = assets)

### multi time steps (in our case 2)
aot_cube_multi<- extract_values_multi(bbox = bbox, crop_box = aot_test, assets = assets )
aoi_cube_multi <- extract_values_multi(bbox = aoi_target, crop_box = bbox_target_utm, assets = assets)

######################################################################################################

### preprocess training data for one time step
training_data_geojson <- "./train_data/train_dat.geojson"

processed_train_data <- preprocess_training_set(
  training_set = training_data_geojson,
  aot_cube = aot_cube, 
  target_column = "Label",
  multiple_time_steps = FALSE
)

### preprocess training data for multi time steps

processed_train_data_multi <- preprocess_training_set(
  training_set = training_data_geojson,
  aot_cube = aot_cube_multi, 
  target_column = "Label",
  multiple_time_steps = TRUE
)

### preprocess training data for regression and only one time step

train <- "./train_data/train_dat_geom.csv"
processed_train_data_regr <- preprocess_training_set(
  training_set = train,
  aot_cube = aot_cube, 
  target_column = "temperature",
  multiple_time_steps = FALSE,
  is_classification = FALSE, 
  srs_train_data = 4326
)

### preprocess training data for regression and multi time steps
processed_train_data_multi_regr <- preprocess_training_set(
  training_set = train,
  aot_cube = aot_cube_multi, 
  target_column = "temperature",
  multiple_time_steps = TRUE,
  is_classification = FALSE, 
  srs_train_data = 4326
)

### random forest classification
rf_class_one <- ml_fit(mlm_rf_class, processed_train_data, "Label")
rf_class_one_predict <- ml_predict(aoi_cube, rf_class_one)
plot(rf_class_one_predict)

rf_class_multi <- ml_fit(mlm_rf_class, processed_train_data_multi, "Label")
rf_class_multi_prediction <- ml_predict(aoi_cube_multi, rf_class_multi)
plot(rf_class_multi_prediction)

new_names <- c("B02_T1", "B03_T1", "B04_T1", "B08_T1","NDVI_T1", "B02_T2", "B03_T2", "B04_T2", "B08_T2", "NDVI_T2")
save_model_as_onnx(
  model = rf_class_multi, 
  model_type = "random_forest", 
  filepath = "random_forest_multi.onnx",
  predictors = new_names, 
  train_data = processed_train_data_multi, 
  target_column = "Label"
)
### random forest regression
rf_regr_one <- ml_fit(mlm_rf_regr, processed_train_data_regr, "temperature")
rf_regr_one_predict <- ml_predict(aoi_cube, rf_regr_one)
plot(rf_regr_one_predict)

rf_regr_multi <- ml_fit(mlm_rf_regr, processed_train_data_multi_regr, "temperature")
rf_regr_multi_predict <- ml_predict(aoi_cube_multi,rf_regr_multi)
plot(rf_regr_multi_predict)

### svm classification
model_svm_class <- ml_fit(mlm_svm_class, processed_train_data, "Label")
prediction_svm <- ml_predict(aoi_cube, model_svm_class)
plot(prediction_svm)

model_svm_class_multi <- ml_fit(mlm_svm_class, processed_train_data_multi, "Label")
prediction_svm_class_multi <- ml_predict(aoi_cube_multi, model_svm_class_multi)
plot(prediction_svm_class_multi)

### svm regression
model_svm_regr <- ml_fit(mlm_svm_regr, processed_train_data_regr, "temperature")
prediction_svm_regr <- ml_predict(aoi_cube, model_svm_regr)
plot(prediction_svm_regr)

model_svm_regr_multi <- ml_fit(mlm_svm_regr, processed_train_data_multi_regr, "temperature")
prediction_svm_regr_multi <- ml_predict(aoi_cube_multi, model_svm_regr_multi)
plot(prediction_svm_regr_multi)
### xprediction_svm_class_multi### xgboost classification 
model_xgb_class <- ml_fit(mlm_xgb_class, processed_train_data, "Label")
prediction_xgb <- ml_predict(aoi_cube, model_xgb_class)
plot(prediction_xgb)

model_xgb_class_multi <- ml_fit(mlm_xgb_class, processed_train_data_multi, "Label")
prediction_xgb_multi <- ml_predict(aoi_cube_multi, model_xgb_class_multi)
plot(prediction_xgb_multi)
### tempcnn 
model_dl <- ml_fit(mlm_tempcnn, processed_train_data, "Label")
prediction_tempcnn <- ml_predict(aoi_cube, model_dl)
plot(prediction_tempcnn)

model_dl_multi <- ml_fit(mlm_tempcnn, processed_train_data_multi, "Label")
prediction_dl_multi <- ml_predict(aoi_cube_multi, model_dl_multi)
plot(prediction_dl_multi)
#####################################################





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
random_forest_model <- load_and_check_onnx_model("random_forest_multi.onnx")
svm_model <- load_and_check_onnx_model("svm.onnx")
xgboost_model <- load_and_check_onnx_model("xgboost_model.onnx")


# Funktion zum Extrahieren der Werte aus dem Cube
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


predict_with_auto_time_step_detection <- function(onnx_model_path, aoi_cube, predictors = NULL, select_bands = NULL, new_names = NULL) {
  # Prüfen der Anzahl der Zeitschritte im Cube
  time_step_count <- gdalcubes::dimensions(aoi_cube)$t$count
  onnxruntime <- reticulate::import("onnxruntime")
  session <- onnxruntime$InferenceSession(onnx_model_path)
  input_name <- session$get_inputs()[[1]]$name
  model_expected_dims <- session$get_inputs()[[1]]$shape[2]
  
  if (time_step_count > 1) {
    # Mehrere Zeitschritte erkannt - Mehrschrittvorhersage
    message("Mehrere Zeitschritte erkannt - Mehrschrittvorhersage wird durchgeführt.")
    
    # Daten für Mehrschrittvorhersage vorbereiten
    time_steps <- time_steps_query(aoi_cube_multi)  # Extrahiere die Zeitschritte aus dem Cube
    combined_data <- prepare_and_combine_multitemporal_cube(aoi_cube, select_bands, time_steps)
    combined_data <- clean_and_rename_data(combined_data, new_names)
    
    # Filter für alle Zeitschritte erstellen und in ein numpy-Array konvertieren
    input_data_list <- list()
    for (time_step in unique(gsub(".*_(T[0-9]+)$", "\\1", names(combined_data)))) {
      time_step_columns <- grep(paste0("_", time_step, "$"), names(combined_data), value = TRUE)
      input_data_list[[time_step]] <- combined_data[, time_step_columns]
    }
    
    input_data <- do.call(cbind, input_data_list)
    input_matrix <- as.matrix(input_data)
    input_matrix <- reticulate::np_array(input_matrix, dtype = "float32")
    
  } else {
    # Ein Zeitschritt erkannt - Einzelschrittvorhersage
    message("Ein Zeitschritt erkannt - Einzelschrittvorhersage wird durchgeführt.")
    
    if (is.null(predictors)) {
      stop("Für eine Einzelschrittvorhersage müssen 'predictors' angegeben werden.")
    }
    
    # Eingabematrix für den einzelnen Zeitschritt
    extracted_data <- extract_cube(cube = aoi_cube, predictors = predictors)
    input_matrix <- extracted_data$data
    cube_dims <- extracted_data$dims
    
  }
  
  # ONNX-Model laden und Vorhersage durchführen
  input_feed <- list(float_input = input_matrix)
  tryCatch({
    outputs <- session$run(NULL, input_feed)
    predictions <- outputs[[1]]
    message("Vorhersage erfolgreich durchgeführt.")
  }, error = function(e) {
    stop("Fehler bei der Vorhersage: ", e$message)
  })
  
  # Plotten für Einzelschrittvorhersage oder Mehrschrittvorhersage
  if (time_step_count == 1) {
    # Einzelschrittvorhersage: Reshape das Ergebnis basierend auf den Cube-Dimensionen
    nrow_cube <- cube_dims[3]
    ncol_cube <- cube_dims[4]
    pred_array <- matrix(predictions, nrow = nrow_cube, ncol = ncol_cube)
    prediction_raster <- terra::rast(pred_array, crs = gdalcubes::srs(aoi_cube))
    
    # Plotten des Einzelschritt-Rasters
    plot(prediction_raster, main = "Vorhersage-Raster (Einzelschritt)", col = terrain.colors(6))
    message("Vorhersage-Raster für Einzelschritt erfolgreich geplottet.")
    
  } else {
    message("Mehre Zeitschritte werden dargestellt")
    # Mehrschrittvorhersage: Verwende die plot_predictions-Funktion
    srs <- gdalcubes::srs(aoi_cube)
    prediction_raster <- create_raster_from_cube(aoi_cube, predictions, srs)
    
    # Überprüfe, ob das Raster korrekt erstellt wurde
    print(prediction_raster)
    
    # Plotte das Raster
    plot(prediction_raster, main = "Vorhersage-Raster", col = terrain.colors(6))  # Anpassung der Farben für 6 Klassen
    message("Vorhersage-Raster erfolgreich geplottet.")
  }
  
  return(predictions)
}


clean_and_rename_data <- function(data_frame, new_names) {
  # Entferne Spalten mit fehlenden Werten
  cleaned_data <- data_frame[, colSums(is.na(data_frame)) == 0]
  
  # Überprüfe, ob die Anzahl der neuen Namen mit der Anzahl der Spalten im bereinigten DataFrame übereinstimmt
  if (length(new_names) == ncol(cleaned_data)) {
    names(cleaned_data) <- new_names
  } else {
    stop("Die Anzahl der neuen Spaltennamen stimmt nicht mit der Anzahl der Spalten im bereinigten DataFrame überein.")
  }
  
  return(cleaned_data)
}

# Parameter und Funktionsaufruf
select_bands <- c("B02", "B03", "B04", "B08")  # Auswahl der Bänder, z.B. Sentinel-Bänder
model_path <- "xgboost.onnx"  # Pfad zum gespeicherten ONNX-Modell
new_names <- c("B02_T1", "B03_T1", "B04_T1", "B08_T1", "B02_T2", "B03_T2", "B04_T2", "B08_T2")  # Benennung der Zeitschritte

predictions <- predict_with_auto_time_step_detection(model_path, aoi_cube = aoi_cube, predictors = select_bands,  select_bands = NULL, new_names = NULL)
predictions <- predict_with_auto_time_step_detection(model_path, aoi_cube = aoi_cube_multi, predictors = NULL,  select_bands = select_bands, new_names = new_names)
print(predictions)



############################# mlm-stac##################################
# Vorhersageraster als GeoTIFF speichern (kein "format" erforderlich)
terra::writeRaster(prediction_raster_rf, filename = "prediction_raster_rf.tif", overwrite = TRUE)
# Laden des gespeicherten GeoTIFF
loaded_raster <- terra::rast("prediction_raster_rf.tif")
loaded_raster

# Plotten des geladenen Rasters
plot(loaded_raster)
library(jsonlite)
library(sf)
library(raster)

# Funktion zur Entfernung von NULL-Werten aus einer Liste
remove_nulls <- function(x) {
  x[!sapply(x, is.null)]
}

generate_stac_metadata <- function(
    model_path,
    prediction_path,
    model,
    data_cube,
    loaded_raster,
    description,
    classes,
    collection_id = "ml-model-examples",
    mlm_name = "RandomForest Sentinel-2 Classification",
    mlm_architecture = "randomForest",
    mlm_tasks = list("classification"),
    mlm_framework = "randomForest",
    mlm_framework_version = as.character(packageVersion("randomForest")),
    mlm_memory_size = NULL,
    mlm_total_parameters = length(model$forest),
    mlm_pretrained = FALSE,
    mlm_pretrained_source = NULL,
    mlm_batch_size_suggestion = NULL,
    mlm_accelerator = NULL,
    mlm_accelerator_constrained = NULL,
    mlm_accelerator_summary = NULL,
    mlm_accelerator_count = NULL,
    mlm_input = NULL,
    mlm_output = list(
      list(
        name = "classification",
        tasks = list("classification"),
        result = list(
          shape = list(-1, length(classes)),
          dim_order = list("batch", "class"),
          data_type = "float32"
        ),
        "classification:classes" = classes
      )
    ),
    mlm_hyperparameters = NULL
) {
  # Extrahiere Bounding Box und CRS
  bbox <- terra::ext(loaded_raster)
  
  # Extrahiere die Bänder und Shape-Informationen aus dem Datacube
  bands <- gdalcubes::bands(data_cube)
  shape <- dim(gdalcubes::as_array(data_cube))  # Liefert das Shape als (Band, Zeit, Höhe, Breite)
  
  # Setzt die Eingabeform basierend auf den Shape-Informationen des Cubes
  if (is.null(mlm_input)) {
    mlm_input <- list(
      list(
        name = "Sentinel-2 Bands",
        bands = bands,
        input = list(
          shape = list(-1, shape[1], shape[3], shape[4]),
          dim_order = list("batch", "channel", "height", "width"),
          data_type = "float32"
        )
      )
    )
  }
  
  # Geometrie im GeoJSON-kompatiblen Format
  geometry <- list(
    type = "Polygon",
    coordinates = list(
      list(
        c(bbox$xmin, bbox$ymin),
        c(bbox$xmin, bbox$ymax),
        c(bbox$xmax, bbox$ymax),
        c(bbox$xmax, bbox$ymin),
        c(bbox$xmin, bbox$ymin)
      )
    )
  )
  
  # Erstellung der STAC-MLM-Metadatenstruktur
  mlm_metadata <- list(
    stac_version = "1.0.0",
    stac_extensions = list(
      "https://stac-extensions.github.io/file/v2.1.0/schema.json",
      "https://crim-ca.github.io/mlm-extension/v1.2.0/schema.json"
    ),
    id = paste0(collection_id, "_item"),
    type = "Feature",
    geometry = geometry,
    bbox = c(bbox$xmin, bbox$ymin, bbox$xmax, bbox$ymax),
    properties = remove_nulls(list(
      start_datetime = Sys.Date(),
      end_datetime = Sys.Date(),
      description = description,
      "mlm:name" = mlm_name,
      "mlm:architecture" = mlm_architecture,
      "mlm:tasks" = mlm_tasks,
      "mlm:framework" = mlm_framework,
      "mlm:framework_version" = mlm_framework_version,
      "mlm:memory_size" = mlm_memory_size,
      "mlm:total_parameters" = mlm_total_parameters,
      "mlm:pretrained" = mlm_pretrained,
      "mlm:pretrained_source" = mlm_pretrained_source,
      "mlm:batch_size_suggestion" = mlm_batch_size_suggestion,
      "mlm:accelerator" = mlm_accelerator,
      "mlm:accelerator_constrained" = mlm_accelerator_constrained,
      "mlm:accelerator_summary" = mlm_accelerator_summary,
      "mlm:accelerator_count" = mlm_accelerator_count,
      "mlm:input" = mlm_input,
      "mlm:output" = mlm_output,
      "mlm:hyperparameters" = mlm_hyperparameters,
      datetime = Sys.time()
    )),
    assets = list(
      model = list(
        href = model_path,
        type = "application/octet-stream",
        title = "RandomForest model",
        description = "RandomForest model trained on Sentinel-2 data",
        roles = list("mlm:model", "mlm:weights", "data")
      ),
      predictions = list(
        href = prediction_path,
        type = "image/tiff",
        title = "Prediction GeoTIFF",
        description = "Predictions of the model on the area of interest.",
        roles = list("mlm:output", "data")
      )
    ),
    links = list(
      list(
        rel = "self",
        href = paste0(collection_id, "_metadata.json"),
        type = "application/json"
      )
    ),
    collection = collection_id
  )
  
  # Speichern der Metadaten als JSON-Datei ohne NULL-Werte
  metadata_json <- toJSON(mlm_metadata, pretty = TRUE, auto_unbox = TRUE)
  json_path <- paste0(collection_id, "_metadata.json")
  write(metadata_json, file = json_path)
  message("STAC-MLM Metadaten erfolgreich gespeichert unter ", json_path)
}

# Beispielaufruf der Funktion nach einem Trainings- oder Vorhersageschritt
model_path <- "random_forest_model.rds"
prediction_path <- "prediction_raster_rf.tif"
description <- "RandomForest classification model trained on Sentinel-2 data."
classes <- list(
  list(value = 0, name = "Acker_bepflanzt", description = "Planted Field"),
  list(value = 1, name = "Acker_unbepflanzt", description = "Unplanted Field"),
  list(value = 2, name = "Industrie", description = "Industry"),
  list(value = 3, name = "Stadt", description = "City"),
  list(value = 4, name = "Wald", description = "Forest"),
  list(value = 5, name = "Wasser", description = "Water")
)

# Falls vorhanden, gib zusätzliche optionale Parameter wie folgt an
generate_stac_metadata(
  model_path = model_path,
  prediction_path = prediction_path,
  model = model,
  data_cube = cube,
  loaded_raster = loaded_raster,
  description = description,
  classes = classes,
  mlm_name = "RandomForest Sentinel-2 Classification",
  mlm_architecture = "randomForest",
  mlm_tasks = list("classification"),
  mlm_framework = "randomForest",
  mlm_framework_version = as.character(packageVersion("randomForest")),
  mlm_memory_size = 500000000,  # Beispielwert in Bytes
  mlm_total_parameters = length(model$forest),
  mlm_pretrained = TRUE,
  mlm_pretrained_source = "Sentinel-2 pretraining",
  mlm_batch_size_suggestion = 32,
  mlm_accelerator = "CPU",
  mlm_accelerator_constrained = FALSE,
  mlm_accelerator_summary = "CPU, 8 threads",
  mlm_accelerator_count = 1,
  mlm_hyperparameters = list(
    n_estimators = 100,
    max_depth = 10,
    min_samples_split = 2
  )
)


#######





