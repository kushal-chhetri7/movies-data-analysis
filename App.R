

library(shiny)
library(dplyr)
library(stringr)
library(highcharter)
library(viridis)
library(tidyr)
library(bslib)
library(caret)
library(randomForest)


imdb <- read.csv("imdb_top_1000.csv", stringsAsFactors = FALSE)


ml_data <- imdb %>%
  select(IMDB_Rating, No_of_Votes, Runtime, Genre, Director) %>%
  mutate(
    Runtime = as.numeric(gsub(" min", "", Runtime)),
    No_of_Votes = log1p(No_of_Votes)
  ) %>%
  na.omit()

ml_data <- ml_data %>%
  separate_rows(Genre, sep = ", ") %>%
  mutate(Genre = trimws(Genre)) %>%
  pivot_wider(
    names_from = Genre,
    values_from = Genre,
    values_fn = function(x) 1,
    values_fill = 0,
    names_prefix = "Genre_"
  )

top_directors <- ml_data %>%
  count(Director) %>%
  arrange(desc(n)) %>%
  slice_head(n = 50) %>%
  pull(Director)

ml_data <- ml_data %>%
  mutate(Director = ifelse(Director %in% top_directors, Director, "Other")) %>%
  mutate(Director = as.factor(Director))

set.seed(123)
trainIndex <- createDataPartition(ml_data$IMDB_Rating, p = 0.8, list = FALSE)
train_data <- ml_data[trainIndex, ]
test_data  <- ml_data[-trainIndex, ]

train_control <- trainControl(method = "cv", number = 5, verboseIter = FALSE)
rf_model <- train(
  IMDB_Rating ~ .,
  data      = train_data,
  method    = "rf",
  trControl = train_control,
  tuneGrid  = data.frame(mtry = c(2, 4, 6)),
  ntree     = 100,
  metric    = "RMSE"
)

get_rating_level <- function(rating) {
  if (rating >= 9)  return("Excellent")
  if (rating >= 8)  return("Great")
  if (rating >= 7)  return("Good")
  return("Others")
}

all_genres <- sort(unique(unlist(strsplit(paste(imdb$Genre, collapse = ", "), ", "))))

ui <- fluidPage(
  title = "Movie Analysis",
  theme = bs_theme(
    version   = 5,
    bootswatch = "flatly",
    base_font = font_google("Roboto"),
    primary   = "#05b373"
  ),
  tags$head(tags$style(HTML("
    body { background: #16171c; }
    .main-panel { background: #16171c; }
    .chart-container { 
      background: #212226; 
      border-radius: 12px; 
      padding: 10px; 
      margin-bottom: 15px;
    }
    .summary-box {
      background: #212226;
      border-radius: 12px;
      padding: 15px;
      margin-bottom: 10px;
      color: #ffffff;
      font-size: 18px;
      text-align: center;
    }
    .summary-title {
      font-size: 14px;
      color: #b0b0b0;
    }
    .summary-value {
      font-size: 24px;
      font-weight: bold;
      color: #05b373;
    }
    .well, .panel, .form-control { border-radius: 6px; }
    .highcharts-title, .highcharts-subtitle {
      font-family: 'Roboto', sans-serif;
      font-weight: 500;
      color: #ffffff !important;
    }
    .title-panel div {
      color: #ffffff;
      font-family: 'Roboto', sans-serif;
      font-size: 32px;
      font-weight: 600;
      text-align: center;
      padding-bottom: 20px;
    }
  ")),
            tags$link(rel = "icon", type = "image/x-icon",
                      href = "https://i.pinimg.com/736x/16/b7/8a/16b78a6ae716b751b0f2d02b2516b04b.jpg")
  ),
  
  titlePanel(
    div(
      "IMDb Movie Analysis Dashboard",
      style = "color: #ffffff; font-family: 'Roboto', sans-serif; font-size: 32px; font-weight: 600; text-align: center; padding-bottom: 20px;"
    )
  ),
  
  sidebarLayout(
    sidebarPanel(
      selectizeInput(
        "genreInput", "Select Genre:",
        choices   = all_genres,
        selected  = "Drama",
        options   = list(
          placeholder = "Type or select a genre...",
          maxOptions  = 30
        )
      ),
      sliderInput(
        "ratingInput", "IMDB Rating:",
        min = 7, max = 10, value = c(8, 10), step = 0.1
      ),
      br(),
      h4("Predict IMDB Rating", style = "color: #ffffff; font-weight: bold;"),
      numericInput(
        "votesInput", "Number of Votes:",
        value = 100000, min = 0, step = 1000
      ),
      numericInput(
        "runtimeInput", "Runtime (minutes):",
        value = 120, min = 0, step = 1
      ),
      selectizeInput(
        "genrePredictInput", "Select Genres (up to 3):",
        choices  = all_genres,
        multiple = TRUE,
        options  = list(maxItems = 3, placeholder = "Select up to 3 genres...")
      ),
      selectizeInput(
        "directorInput", "Select Director:",
        choices   = c("Other", top_directors),
        selected  = "Other",
        options   = list(placeholder = "Type or select a director...")
      ),
      actionButton("predictButton", "Predict Rating", class = "btn-primary"),
      br(), br(),
      div(class = "summary-box",
          div(class = "summary-title", "Predicted IMDB Rating"),
          div(class = "summary-value", textOutput("predictedRating"))
      ),
      br(),
      fluidRow(
        column(6, div(class = "summary-box",
                      div(class = "summary-title", "Total Movies"),
                      div(class = "summary-value", textOutput("totalMovies"))
        )),
        column(6, div(class = "summary-box",
                      div(class = "summary-title", "Average Rating"),
                      div(class = "summary-value", textOutput("avgRating"))
        ))
      ),
      style = "background-color: #212226; color: #ffffff; border-radius: 8px; margin-bottom: 12px; max-height: 855.5px; overflow-y: auto; padding-right: 10px;"
    ),
    
    mainPanel(
      fluidRow(
        column(6, 
               div(class = "chart-container", highchartOutput("barPlot", height = "400px"))
        ),
        column(6, 
               div(class = "chart-container", highchartOutput("linePlot", height = "400px"))
        )
      ),
      fluidRow(
        column(6, 
               div(class = "chart-container", highchartOutput("histPlot", height = "400px"))
        ),
        column(6, 
               div(class = "chart-container", highchartOutput("piePlot", height = "400px"))
        )
      )
    )
  )
)

server <- function(input, output, session) {
  
  genre_filtered <- reactive({
    imdb %>%
      filter(str_detect(Genre, fixed(input$genreInput))) %>%
      filter(IMDB_Rating >= input$ratingInput[1],
             IMDB_Rating <= input$ratingInput[2])
  })
  
  output$totalMovies <- renderText({
    nrow(genre_filtered())
  })
  
  output$avgRating <- renderText({
    avg <- mean(genre_filtered()$IMDB_Rating, na.rm = TRUE)
    if (is.na(avg)) return("N/A")
    round(avg, 2)
  })
  
  output$barPlot <- renderHighchart({
    top_directors <- genre_filtered() %>%
      group_by(Director) %>%
      summarize(n = n()) %>%
      arrange(desc(n)) %>%
      slice_head(n = 10)
    
    custom_colors <- viridis::viridis(10, option = "D", end = 0.85)
    
    highchart() %>%
      hc_chart(type = "bar", backgroundColor = "#212226") %>%
      hc_add_series(
        data = top_directors$n,
        name = "Number of Movies",
        colorByPoint = TRUE,
        colors = custom_colors,
        borderColor = "none",
        borderWidth = 0
      ) %>%
      hc_xAxis(
        categories = top_directors$Director,
        title = list(text = "Director", style = list(color = "#ffffff")),
        labels = list(style = list(color = "#ffffff")),
        gridLineColor = "#212226"
      ) %>%
      hc_yAxis(
        labels = list(style = list(color = "#ffffff")),
        gridLineColor = "#333333"
      ) %>%
      hc_title(text = paste("Top Directors in", input$genreInput),
               style = list(color = "white")) %>%
      hc_add_theme(hc_theme_flat())
  })
  
  output$linePlot <- renderHighchart({
    req(nrow(genre_filtered()) > 0)
    avg_rating_year <- genre_filtered() %>%
      filter(!is.na(Released_Year)) %>%
      group_by(Released_Year) %>%
      summarize(Avg_Rating = mean(IMDB_Rating, na.rm = TRUE)) %>%
      filter(!is.na(Avg_Rating))
    
    highchart() %>%
      hc_chart(type = "line", backgroundColor = "#212226") %>%
      hc_add_series(
        data = avg_rating_year$Avg_Rating,
        name = "Average Rating",
        color = "#00b374"
      ) %>%
      hc_xAxis(
        categories = avg_rating_year$Released_Year,
        title = list(text = "Year", style = list(color = "#ffffff")),
        labels = list(style = list(color = "#ffffff")),
        gridLineColor = "#333333"
      ) %>%
      hc_yAxis(
        labels = list(style = list(color = "#ffffff")),
        gridLineColor = "#212226"
      ) %>%
      hc_title(text = paste("Yearly Avg Rating for", input$genreInput),
               style = list(color = "white")) %>%
      hc_add_theme(hc_theme_flat())
  })
  
  output$histPlot <- renderHighchart({
    req(nrow(genre_filtered()) > 0)
    ratings_for_hist <- genre_filtered()$IMDB_Rating[!is.na(genre_filtered()$IMDB_Rating)]
    req(length(ratings_for_hist) > 0 && var(ratings_for_hist, na.rm = TRUE) > 0)
    
    breaks <- seq(floor(min(ratings_for_hist, na.rm=TRUE)),
                  ceiling(max(ratings_for_hist, na.rm=TRUE)), by = 0.2)
    if(length(breaks) < 2) {
      breaks <- c(floor(min(ratings_for_hist, na.rm=TRUE)),
                  ceiling(max(ratings_for_hist, na.rm=TRUE)))
    }
    hist_data <- hist(ratings_for_hist, breaks = breaks, plot = FALSE)
    custom_hist_colors <- viridis::viridis(length(hist_data$counts),
                                           option = "D",
                                           end = 0.85)
    
    highchart() %>%
      hc_chart(type = "column", backgroundColor = "#212226") %>%
      hc_add_series(
        data = hist_data$counts,
        name = "Count",
        colorByPoint = TRUE,
        colors = custom_hist_colors,
        borderColor = "none",
        borderWidth = 0,
        pointPadding = 0,
        groupPadding = 0
      ) %>%
      hc_xAxis(
        categories = round(hist_data$mids, 2),
        title = list(text = "IMDB Rating", style = list(color = "#ffffff")),
        labels = list(style = list(color = "#ffffff")),
        gridLineColor = "#333333"
      ) %>%
      hc_yAxis(
        title = list(text = "Count", style = list(color = "#ffffff")),
        labels = list(style = list(color = "#ffffff")),
        gridLineColor = "#212226"
      ) %>%
      hc_title(text = paste("Rating Distribution for", input$genreInput),
               style = list(color = "white")) %>%
      hc_add_theme(hc_theme_flat())
  })
  
  output$piePlot <- renderHighchart({
    req(nrow(genre_filtered()) > 0)
    neon_cols <- c(
      "Excellent" = "#cc353c",
      "Great"     = "#5e0414",
      "Good"      = "#c701ae",
      "Others"    = "#747474"
    )
    
    dist <- genre_filtered() %>%
      mutate(Rating_Level_App = sapply(IMDB_Rating, get_rating_level)) %>%
      count(Rating_Level_App, name = "y") %>%
      arrange(factor(Rating_Level_App, levels = names(neon_cols))) %>%
      mutate(
        name  = Rating_Level_App,
        color = neon_cols[Rating_Level_App]
      ) %>%
      filter(!is.na(color))
    
    highchart() %>%
      hc_chart(type = "pie", backgroundColor = "#212226") %>%
      hc_add_series(
        data = list_parse(dist),
        colorByPoint = TRUE,
        borderColor  = "none",
        borderWidth  = 0
      ) %>%
      hc_title(
        text = paste("Rating Level Distribution in", input$genreInput),
        style = list(color = "white")
      ) %>%
      hc_add_theme(hc_theme_flat())
  })
  

  output$predictedRating <- renderText({

    input$predictButton
    
    isolate({
      req(input$votesInput, input$runtimeInput, input$genrePredictInput, input$directorInput)
      
      new_data <- data.frame(
        No_of_Votes = log1p(as.numeric(input$votesInput)),
        Runtime     = as.numeric(input$runtimeInput),
        Director    = input$directorInput,
        stringsAsFactors = FALSE
      )
      

      genre_cols <- colnames(ml_data)[grepl("^Genre_", colnames(ml_data))]
      new_data[genre_cols] <- 0
      
      selected_genres <- input$genrePredictInput
      valid_genres    <- intersect(selected_genres, all_genres)
      
      for (genre in valid_genres) {
        genre_col <- paste0("Genre_", trimws(genre))
        if (genre_col %in% genre_cols) {
          new_data[[genre_col]] <- 1
        }
      }
      
      new_data$Director <- factor(new_data$Director, levels = levels(ml_data$Director))
      
      missing_cols <- setdiff(colnames(ml_data), c("IMDB_Rating", colnames(new_data)))
      if (length(missing_cols) > 0) {
        new_data[missing_cols] <- 0
      }
      
      new_data <- new_data[, colnames(ml_data)[!colnames(ml_data) %in% "IMDB_Rating"]]
      
      prediction <- predict(rf_model, new_data)
      sprintf("%.2f", prediction)
    })
  })
}

shinyApp(ui, server)