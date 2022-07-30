/*
    An example sketch for Edge Impulse trained model inference for Anomaly detection with Accelerometer and data upload to BLynk Edgent

    Copyright (c) 2021 Universiti Malaysia Perlis
    Author      : Muhammad Amri bin Hassan
    Create Time : July 2022
    Change Log  :

    The MIT License (MIT)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
*/
#define MIN_CONFIDENCE 0.8
#define ANOMALY_THRESHOLD 3

#define BLYNK_TEMPLATE_ID "TMPLo5QFa2nK"
#define BLYNK_DEVICE_NAME "Wio"
#define BLYNK_AUTH_TOKEN "ihXuEtuEFH3fTLiuUhk09U9LyomVwjvl"

#define BLYNK_FIRMWARE_VERSION        "0.1.0"
#define BLYNK_PRINT Serial
#define APP_DEBUG

/* Includes ---------------------------------------------------------------- */

#include "BlynkEdgent.h"
#include <Anomaly_detection_inferencing.h>
#include"LIS3DHTR.h"
#include <Seeed_Arduino_FreeRTOS.h>
#include"TFT_eSPI.h"

TFT_eSPI tft;
LIS3DHTR<TwoWire> lis;

/* Constant defines -------------------------------------------------------- */
#define CONVERT_G_TO_MS2    9.80665f

/* Private variables ------------------------------------------------------- */
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
uint8_t best_result = 0;
/**
* @brief      Arduino setup function
*/
void setup()
{   
    // put your setup code here, to run once:
    pinMode(WIO_BUZZER, OUTPUT);
    Serial.begin(115200);

    #ifdef APP_DEBUG 
    while (!Serial) {delay(10);}
    #endif

    Serial.println("Edge Impulse Inferencing Demo");

    tft.begin();
    tft.setRotation(3);
    
    lis.begin(Wire1);
 
    if (!lis.available()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
    }
    else {
        ei_printf("IMU initialized\r\n");
    }
    lis.setOutputDataRate(LIS3DHTR_DATARATE_100HZ); // Setting output data rage to 25Hz, can be set up tp 5kHz 
    lis.setFullScaleRange(LIS3DHTR_RANGE_16G); // Setting scale range to 2g, select from 2,4,8,16g
    

    if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 3) {
        ei_printf("ERR: EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME should be equal to 3 (the 3 sensor axes)\n");
        return;
    }

    BlynkEdgent.begin();
    timer.setInterval(1000L, send_data);
}

/**
* @brief      Printf function uses vsnprintf and output using Arduino Serial
*
* @param[in]  format     Variable argument list
*/
void ei_printf(const char *format, ...) {
   static char print_buf[1024] = { 0 };

   va_list args;
   va_start(args, format);
   int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
   va_end(args);

   if (r > 0) {
       Serial.write(print_buf);
   }
}

void run_inference()
{
    ei_printf("Sampling...\n");

    // Allocate a buffer here for the values we'll read from the IMU
    float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
        // Determine the next tick (and then sleep later)
        uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

        //IMU.readAcceleration(buffer[ix], buffer[ix + 1], buffer[ix + 2]);
        lis.getAcceleration(&buffer[ix], &buffer[ix+1], &buffer[ix + 2]);
        buffer[ix + 0] *= CONVERT_G_TO_MS2;
        buffer[ix + 1] *= CONVERT_G_TO_MS2;
        buffer[ix + 2] *= CONVERT_G_TO_MS2;

        delayMicroseconds(next_tick - micros());
    }

    // Turn the raw buffer in a signal which we can the classify
    signal_t signal;
    int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    if (err != 0) {
        ei_printf("Failed to create signal from buffer (%d)\n", err);
        return;
    }

    // Run the classifier
    ei_impulse_result_t result = {0};

    err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        return;
    }

    // print the predictions
    ei_printf("Predictions ");
    ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
        result.timing.dsp, result.timing.classification, result.timing.anomaly);
    ei_printf(": \n");

    float prev_best = 0.0;
    
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
        if (result.classification[ix].value > prev_best)
        {
        prev_best = result.classification[ix].value;
        best_result = ix;
        }
    }
    
  #if EI_CLASSIFIER_HAS_ANOMALY == 1
      ei_printf("    anomaly score: %.3f\n", result.anomaly);
      if (result.anomaly > ANOMALY_THRESHOLD)
        {       
        best_result = EI_CLASSIFIER_LABEL_COUNT;
        Blynk.logEvent("anomaly_alert","Anomaly Detected!");

         tft.fillScreen(TFT_RED);
         tft.setFreeFont(&FreeSansBoldOblique12pt7b);
         tft.drawString("Anomaly detected", 20, 80);
         delay(1000);
         tft.fillScreen(TFT_WHITE);
         
          analogWrite(WIO_BUZZER, 128);
          delay(1000);
          analogWrite(WIO_BUZZER, 0);
          delay(1000);
         
        }
  #endif

   //ei_printf("Best prediction %d \n", best_result);
  
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_ACCELEROMETER
#error "Invalid model for current sensor"
#endif

void send_data()
{
  uint16_t noise = analogRead(WIO_MIC);
  uint16_t light = analogRead(WIO_LIGHT);

  String str = String(noise);
  Blynk.virtualWrite(V0, str);
  Serial.println(str);
  str = String(light);
  Blynk.virtualWrite(V1, str);
  Serial.println(str); 
  ei_printf("Best prediction %d \n", best_result);
  Blynk.virtualWrite(V2, best_result);
}


/**
* @brief      Get data and run inferencing
*
* @param[in]  debug  Get debug info if true
*/
void loop()
{
    run_inference();
    BlynkEdgent.run(); 
}
