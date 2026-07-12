plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.josemrm.lesho_app"
    compileSdk = 36
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "com.josemrm.lesho_app"
        // MediaPipe Tasks Vision requiere minSdk 24. Se fija explicito porque el
        // Modelo A corre con landmarks de MediaPipe on-device.
        minSdk = 24
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    // No comprimir los modelos: MediaPipe (.task) y TFLite (.tflite) se leen
    // directamente del asset; comprimirlos rompe la carga por mapeo de memoria.
    androidResources {
        noCompress.add("task")
        noCompress.add("tflite")
    }

    buildTypes {
        release {
            // Firmado con las llaves de debug por ahora (no se publica en tienda).
            signingConfig = signingConfigs.getByName("debug")
            // R8 ofusca las clases protobuf de MediaPipe y rompe su reflexión
            // ("Field platform_ ... not found"), así que HandLandmarker no inicia.
            // Se desactiva la minificación en release para que funcione igual que
            // en debug. (Si algún día se quiere optimizar el tamaño, activar con
            // reglas -keep para com.google.mediapipe.** y com.google.protobuf.**.)
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }
}

dependencies {
    // MediaPipe Tasks Vision: HandLandmarker on-device (extrae los 21 puntos por
    // mano). Es el mismo motor y modelo (hand_landmarker.task) que el pipeline de
    // entrenamiento en Python, para que la app vea exactamente lo que vio el modelo.
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
