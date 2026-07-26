package com.josemrm.lesho_app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.ByteArrayOutputStream

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
                    "liberar" -> handlerDeteccion.post {
                        hands?.close(); hands = null
                        pose?.close(); pose = null
                        handlerPrincipal.post { result.success(true) }
                    }
                    else -> result.notImplemented()
                }
            }
    }

    private fun inicializar(numManos: Int, conPose: Boolean) {
        hands?.close(); hands = null
        pose?.close(); pose = null
        val t0 = System.currentTimeMillis()
        hands = try {
            crearHands(numManos, Delegate.GPU)
        } catch (e: Exception) {
            crearHands(numManos, Delegate.CPU)
        }
        val t1 = System.currentTimeMillis()
        android.util.Log.d("LESHO_INIT", "Hands: ${t1 - t0} ms")
        if (conPose) {
            pose = try {
                crearPose(Delegate.GPU)
            } catch (e: Exception) {
                crearPose(Delegate.CPU)
            }
            android.util.Log.d("LESHO_INIT", "Pose: ${System.currentTimeMillis() - t1} ms")
        }
        timestampMs = 0
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

    private fun detectar(
        nv21: ByteArray, width: Int, height: Int, rotation: Int, conPoseFrame: Boolean
    ): Map<String, Any?> {
        val h = hands ?: return mapOf("manos" to emptyList<Any>(), "pose" to null)

        // NV21 -> JPEG -> Bitmap -> rotar vertical + espejo (selfie).
        val yuv = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val jpegStream = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, width, height), 85, jpegStream)
        val jpeg = jpegStream.toByteArray()
        var bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
            ?: return mapOf("manos" to emptyList<Any>(), "pose" to null)
        val m = Matrix()
        m.postRotate(rotation.toFloat())
        m.postScale(-1f, 1f)
        bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)

        val mpImage: MPImage = BitmapImageBuilder(bmp).build()
        timestampMs += 33

        val manos = detectarManos(h, mpImage, timestampMs)
        val puntosPose = if (conPoseFrame) {
            pose?.let { detectarPose(it, mpImage, timestampMs) }
        } else {
            null
        }

        return mapOf("manos" to manos, "pose" to puntosPose)
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
