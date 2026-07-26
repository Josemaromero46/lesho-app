package com.josemrm.lesho_app

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * Puente nativo con MediaPipe (Tasks API): manos (Modelo A y B) y, opcionalmente,
 * cuerpo/Pose (Modelo B). Mismo motor y mismos modelos que el pipeline de Python.
 *
 * Toda la detección corre en un HILO DE FONDO para no congelar el preview. La
 * respuesta al canal se entrega en el hilo principal, como exige Flutter.
 *
 * Canal "lesho/manos":
 *   - inicializar(numManos, conPose): crea el HandLandmarker (1 o 2 manos) y, si
 *     conPose, el PoseLandmarker (para la ubicación en el cuerpo del Modelo B).
 *   - detectar(bytes NV21, width, height, rotation) -> mapa:
 *       "manos": lista de manos {lateralidad, score, landmarks[63]}
 *       "pose" : 33 puntos [x0,y0,z0,vis0, ...] (132 valores) o null si no hay pose.
 *   - liberar()
 */
class MainActivity : FlutterActivity() {
    private val canal = "lesho/manos"
    private var hands: HandLandmarker? = null
    private var pose: PoseLandmarker? = null
    private var numManosActual = -1
    private var timestampMs: Long = 0

    private val hiloDeteccion = HandlerThread("mediapipe").apply { start() }
    private val handlerDeteccion = Handler(hiloDeteccion.looper)
    private val handlerPrincipal = Handler(Looper.getMainLooper())

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, canal)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "inicializar" -> {
                        val numManos = call.argument<Int>("numManos") ?: 1
                        val conPose = call.argument<Boolean>("conPose") ?: false
                        handlerDeteccion.post {
                            try {
                                inicializar(numManos, conPose)
                                handlerPrincipal.post { result.success(true) }
                            } catch (e: Exception) {
                                handlerPrincipal.post { result.error("INIT", e.message, null) }
                            }
                        }
                    }
                    "detectar" -> {
                        val bytes = call.argument<ByteArray>("bytes")
                        val w = call.argument<Int>("width")
                        val h = call.argument<Int>("height")
                        val rot = call.argument<Int>("rotation") ?: 0
                        // Pose es opcional POR FOTOGRAMA: el cuerpo es estable, así que
                        // basta correrla cada pocos frames y reusar el último marco.
                        val conPoseFrame = call.argument<Boolean>("conPose") ?: true
                        if (bytes == null || w == null || h == null) {
                            result.error("DETECT", "argumentos invalidos", null)
                        } else {
                            handlerDeteccion.post {
                                try {
                                    val salida = detectar(bytes, w, h, rot, conPoseFrame)
                                    handlerPrincipal.post { result.success(salida) }
                                } catch (e: Exception) {
                                    handlerPrincipal.post { result.error("DETECT", e.message, null) }
                                }
                            }
                        }
                    }
                    "detectarLote" -> {
                        val crudos = call.argument<List<Any>>("frames")
                        val w = call.argument<Int>("width")
                        val h = call.argument<Int>("height")
                        val rot = call.argument<Int>("rotation") ?: 0
                        val poseCada = call.argument<Int>("poseCada") ?: 3
                        if (crudos == null || w == null || h == null) {
                            result.error("LOTE", "argumentos invalidos", null)
                        } else {
                            handlerDeteccion.post {
                                try {
                                    val frames = crudos.map { it as ByteArray }
                                    val salida = detectarLote(frames, w, h, rot, poseCada)
                                    handlerPrincipal.post { result.success(salida) }
                                } catch (e: Exception) {
                                    handlerPrincipal.post { result.error("LOTE", e.message, null) }
                                }
                            }
                        }
                    }
                    "liberar" -> {
                        // Las instancias se CONSERVAN a propósito: crearlas cuesta
                        // ~12 s (Hands en GPU). Al volver a entrar a una pantalla,
                        // "inicializar" las reusa y la carga es casi instantánea.
                        // Se cierran de verdad solo en onDestroy.
                        result.success(true)
                    }
                    else -> result.notImplemented()
                }
            }
    }

    // DELEGADOS MEDIDOS EN EL A13 (logs LESHO_INIT / LESHO_T, 2026-07-26):
    //   Hands GPU: init ~12 s, inferencia ~250 ms/frame.
    //   Hands CPU: init ~0.7 s, inferencia ~550 ms/frame (el doble de lento, y
    //              satura los núcleos que la cámara y la UI necesitan).
    //   Pose GPU:  init ~16 s.  Pose CPU: init ~2.8 s, ~325 ms/corrida.
    // Decisión HÍBRIDA: Hands en GPU (la inferencia manda: corre en vivo en el
    // deletreo) y Pose en CPU (solo corre fuera de línea al clasificar una palabra,
    // y su init GPU de 16 s no se justifica). El costo del init de GPU se paga UNA
    // vez por sesión: las instancias se conservan (ver "liberar") y se reusan.
    private fun inicializar(numManos: Int, conPose: Boolean) {
        val t0 = System.currentTimeMillis()
        if (hands != null && numManos == numManosActual) {
            android.util.Log.d("LESHO_INIT", "Hands: reuso")
        } else {
            hands?.close(); hands = null
            hands = try {
                crearHands(numManos, Delegate.GPU)
            } catch (e: Exception) {
                crearHands(numManos, Delegate.CPU)
            }
            numManosActual = numManos
            android.util.Log.d("LESHO_INIT", "Hands: ${System.currentTimeMillis() - t0} ms")
        }
        if (conPose) {
            if (pose == null) {
                val t1 = System.currentTimeMillis()
                pose = try {
                    crearPose(Delegate.CPU)
                } catch (e: Exception) {
                    crearPose(Delegate.GPU)
                }
                android.util.Log.d("LESHO_INIT", "Pose: ${System.currentTimeMillis() - t1} ms")
            } else {
                android.util.Log.d("LESHO_INIT", "Pose: reuso")
            }
        }
        // El contador de timestamps NUNCA se reinicia: el modo VIDEO exige marcas
        // crecientes y una instancia conservada ya consumió las anteriores.
    }

    private fun crearHands(numManos: Int, delegate: Delegate): HandLandmarker {
        val base = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .setDelegate(delegate)
            .build()
        val opciones = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(numManos)
            .setMinHandDetectionConfidence(0.6f)
            .setMinHandPresenceConfidence(0.6f)
            .setMinTrackingConfidence(0.5f)
            .build()
        return HandLandmarker.createFromOptions(this, opciones)
    }

    private fun crearPose(delegate: Delegate): PoseLandmarker {
        val base = BaseOptions.builder()
            .setModelAssetPath("pose_landmarker.task")
            .setDelegate(delegate)
            .build()
        val opciones = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.VIDEO)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(0.5f)
            .setMinPosePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()
        return PoseLandmarker.createFromOptions(this, opciones)
    }

    /// Convierte NV21 a Bitmap ARGB directamente (sin pasar por JPEG). El camino
    /// viejo comprimía a JPEG y lo descomprimía POR FOTOGRAMA: lento y con pérdida
    /// (el pipeline de entrenamiento en Python nunca pasa por JPEG). Conversión
    /// BT.601 estándar de Android.
    private fun nv21ABitmap(nv21: ByteArray, width: Int, height: Int): Bitmap {
        val frameSize = width * height
        val argb = IntArray(frameSize)
        var yp = 0
        for (j in 0 until height) {
            var uvp = frameSize + (j shr 1) * width
            var u = 0
            var v = 0
            for (i in 0 until width) {
                var y = (nv21[yp].toInt() and 0xFF) - 16
                if (y < 0) y = 0
                if (i and 1 == 0) {
                    v = (nv21[uvp++].toInt() and 0xFF) - 128
                    u = (nv21[uvp++].toInt() and 0xFF) - 128
                }
                val y1192 = 1192 * y
                var r = y1192 + 1634 * v
                var g = y1192 - 833 * v - 400 * u
                var b = y1192 + 2066 * u
                if (r < 0) r = 0 else if (r > 262143) r = 262143
                if (g < 0) g = 0 else if (g > 262143) g = 262143
                if (b < 0) b = 0 else if (b > 262143) b = 262143
                argb[yp] = -0x1000000 or ((r shl 6) and 0xFF0000) or
                    ((g shr 2) and 0xFF00) or ((b shr 10) and 0xFF)
                yp++
            }
        }
        return Bitmap.createBitmap(argb, width, height, Bitmap.Config.ARGB_8888)
    }

    /// Arma el MPImage de un fotograma: conversión directa + rotación vertical +
    /// espejo (selfie). La misma Matrix que usaba el camino viejo, así la geometría
    /// de los landmarks no cambia.
    private fun prepararImagen(
        nv21: ByteArray, width: Int, height: Int, rotation: Int
    ): MPImage {
        var bmp = nv21ABitmap(nv21, width, height)
        val m = Matrix()
        m.postRotate(rotation.toFloat())
        m.postScale(-1f, 1f)
        bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        return BitmapImageBuilder(bmp).build()
    }

    private fun detectar(
        nv21: ByteArray, width: Int, height: Int, rotation: Int, conPoseFrame: Boolean
    ): Map<String, Any?> {
        val h = hands ?: return mapOf("manos" to emptyList<Any>(), "pose" to null)

        val mpImage = prepararImagen(nv21, width, height, rotation)
        timestampMs += 33

        val manos = detectarManos(h, mpImage, timestampMs)
        val puntosPose = if (conPoseFrame) {
            pose?.let { detectarPose(it, mpImage, timestampMs) }
        } else {
            null
        }

        return mapOf("manos" to manos, "pose" to puntosPose)
    }

    /// Procesa un LOTE de fotogramas en una sola llamada del canal (para la
    /// clasificación de una palabra al soltar el botón). Evita un viaje
    /// Dart<->nativo por fotograma. La pose corre cada [poseCada] fotogramas.
    /// Registra los tiempos por etapa en el log (LESHO_T) para diagnóstico.
    private fun detectarLote(
        frames: List<ByteArray>, width: Int, height: Int, rotation: Int, poseCada: Int
    ): List<Map<String, Any?>> {
        val h = hands ?: return emptyList()
        val salidas = ArrayList<Map<String, Any?>>(frames.size)
        var tConv = 0L
        var tHands = 0L
        var tPose = 0L
        val t0 = System.currentTimeMillis()
        for ((indice, nv21) in frames.withIndex()) {
            var t = System.currentTimeMillis()
            val mpImage = prepararImagen(nv21, width, height, rotation)
            tConv += System.currentTimeMillis() - t
            timestampMs += 33

            t = System.currentTimeMillis()
            val manos = detectarManos(h, mpImage, timestampMs)
            tHands += System.currentTimeMillis() - t

            var puntosPose: DoubleArray? = null
            if (indice % poseCada == 0) {
                t = System.currentTimeMillis()
                puntosPose = pose?.let { detectarPose(it, mpImage, timestampMs) }
                tPose += System.currentTimeMillis() - t
            }
            salidas.add(mapOf("manos" to manos, "pose" to puntosPose))
        }
        android.util.Log.d(
            "LESHO_T",
            "lote n=${frames.size} conv=${tConv}ms hands=${tHands}ms " +
                "pose=${tPose}ms total=${System.currentTimeMillis() - t0}ms"
        )
        return salidas
    }

    private fun detectarManos(
        h: HandLandmarker, img: MPImage, ts: Long
    ): List<Map<String, Any>> {
        val res: HandLandmarkerResult = h.detectForVideo(img, ts)
        val manos = ArrayList<Map<String, Any>>()
        val listas = res.landmarks()
        val handed = res.handednesses()
        for (i in listas.indices) {
            val pts = listas[i]
            val coords = DoubleArray(pts.size * 3)
            for (j in pts.indices) {
                coords[j * 3] = pts[j].x().toDouble()
                coords[j * 3 + 1] = pts[j].y().toDouble()
                coords[j * 3 + 2] = pts[j].z().toDouble()
            }
            var lateralidad = "Desconocida"
            var score = 0.0
            if (i < handed.size && handed[i].isNotEmpty()) {
                lateralidad = handed[i][0].categoryName()
                score = handed[i][0].score().toDouble()
            }
            manos.add(hashMapOf(
                "lateralidad" to lateralidad, "score" to score, "landmarks" to coords
            ))
        }
        return manos
    }

    // Devuelve los 33 puntos de la pose como [x,y,z,vis] por punto (132 valores).
    private fun detectarPose(p: PoseLandmarker, img: MPImage, ts: Long): DoubleArray? {
        val res: PoseLandmarkerResult = p.detectForVideo(img, ts)
        val listas = res.landmarks()
        if (listas.isEmpty()) return null
        val pts = listas[0]
        val coords = DoubleArray(pts.size * 4)
        for (j in pts.indices) {
            coords[j * 4] = pts[j].x().toDouble()
            coords[j * 4 + 1] = pts[j].y().toDouble()
            coords[j * 4 + 2] = pts[j].z().toDouble()
            coords[j * 4 + 3] = pts[j].visibility().orElse(1.0f).toDouble()
        }
        return coords
    }

    override fun onDestroy() {
        handlerDeteccion.post {
            hands?.close(); hands = null
            pose?.close(); pose = null
        }
        hiloDeteccion.quitSafely()
        super.onDestroy()
    }
}
