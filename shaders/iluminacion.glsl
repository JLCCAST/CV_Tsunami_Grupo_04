// =====================================================================
//  iluminacion.glsl - Rol 5 (Iluminación, Color y Sombras)
//  Coordinación con Rol 4 (Materiales) para la integración
//
//  Cómo se usa:
//  armar_fragment() (en modulos/iluminacion.py) concatena este archivo con
//  el fragment shader. Aquí solo van las funciones de cálculo; no declaramos
//  in, out ni uniforms para no chocar con tus variables.
//
//  Convenciones rápidas:
//  - Espacio de mundo con eje Y hacia arriba.
//  - N, V y L deben llegar normalizados.
//  - L apunta HACIA el sol (Python lo publica en ctx.sol_dir).
//  - humedad, alarma y visibilidad van de 0.0 a 1.0.
//  - Uniforms que publica mi módulo: uSolDir, uSolColor, uHora, uAlarma.
//
//  ---------------------------------------------------------------------
//  Funciones listas para usar en tu shader de materiales (Rol 4):
//  ---------------------------------------------------------------------
//  1. phong_mojado(N, V, L, color_base, color_luz, ka, kd, ks, brillo, humedad, visibilidad)
//     -> La principal para terreno y casas. Con humedad > 0 oscurece el albedo y sube el brillo especular.
//
//  2. phong(N, V, L, color_base, color_luz, ka, kd, ks, brillo, visibilidad)
//     -> Phong estándar para superficies que se mantengan secas.
//
//  3. sombra(mapa, pos_luz, n_dot_l)
//     -> Devuelve visibilidad (1.0 = iluminado, 0.0 = sombra).
//     *Para la semana 1 puedes pasarle 1.0 fijo mientras integramos el shadow map.
//
//  4. tinte_alarma(color, alarma)
//     -> Aplícalo al final sobre el color ya iluminado para oscurecer y virar a rojo si sube el nivel crítico.
//
//  5. rampa_profundidad(d, d_max)
//     -> Color base del agua según su profundidad W en metros (de turquesa a azul profundo).
//
//  6. rampa_color(t)
//     -> Riesgo de 0 a 1 (verde -> amarillo -> rojo). Usa los mismos colores exactos del HUD del Rol 1.
//
//  ---------------------------------------------------------------------
//  Puntos a coordinar con Rol 4 (coméntame en el PR o por interno):
//  - ¿Te parece bien hacer Phong por fragmento (por píxel) para que la iluminación del agua y casas coincida?
//  - Dejé ka, kd, ks y brillo como parámetros por función para que manejes los valores de cada material a tu gusto.
//  - La textura de humedad la manejas tú y a mis funciones solo me pasas el float ya muestreado en el punto.
//  - Para las sombras de la semana 2, yo me encargo de calcular la matriz de luz para que tú solo pases pos_luz.
// =====================================================================

const float PI = 3.14159265359;

// ---------------------------------------------------------------------
//  SOL: luz direccional con ciclo de día (hora 0..24)
//  Devuelve L (hacia el sol). Amanece en x+ a las 6 h, mediodía arriba.
// ---------------------------------------------------------------------
vec3 direccion_sol(float hora) {
    float ang = (hora - 6.0) / 12.0 * PI;          // 6h -> 0, 12h -> PI/2, 18h -> PI
    return normalize(vec3(cos(ang), sin(ang), 0.35));
}

// Atenúa la luz cuando el sol está bajo el horizonte
float intensidad_sol(vec3 L) {
    return clamp(L.y * 3.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------
//  MODELO DE PHONG (iluminación local)
//    I = ka*Cb  +  V * ( kd*max(N·L,0)*Cb*Cl  +  ks*max(R·V,0)^n*Cl )
//    R = reflect(-L, N).   'visibilidad' (de sombra()) apaga difuso y
//    especular, pero NO el ambiente.
// ---------------------------------------------------------------------
vec3 phong(vec3 N, vec3 V, vec3 L, vec3 color_base, vec3 color_luz,
           float ka, float kd, float ks, float brillo, float visibilidad) {
    float n_dot_l = max(dot(N, L), 0.0);
    vec3 R = reflect(-L, N);
    float r_dot_v = max(dot(R, V), 0.0);

    vec3 ambiente  = ka * color_base;
    vec3 difuso    = kd * n_dot_l * color_base * color_luz;
    vec3 especular = (n_dot_l > 0.0)
                   ? ks * pow(r_dot_v, brillo) * color_luz
                   : vec3(0.0);

    return ambiente + visibilidad * (difuso + especular);
}

// ---------------------------------------------------------------------
//  EFECTO "MOJADO" (usa la textura de humedad que entrega el Rol 4)
//  Mojado = albedo más oscuro + especular fuerte + brillo más concentrado.
// ---------------------------------------------------------------------
vec3 phong_mojado(vec3 N, vec3 V, vec3 L, vec3 color_base, vec3 color_luz,
                  float ka, float kd, float ks, float brillo,
                  float humedad, float visibilidad) {
    vec3  base_mojada = color_base * mix(1.0, 0.55, humedad);
    float ks_m        = mix(ks, 0.90, humedad);
    float brillo_m    = mix(brillo, 128.0, humedad);
    return phong(N, V, L, base_mojada, color_luz, ka, kd, ks_m, brillo_m, visibilidad);
}

// ---------------------------------------------------------------------
//  SOMBRAS: shadow mapping (ocultamiento por imagen) con PCF 3x3
//  pos_luz = matriz_luz * posicion_mundo (clip space de la luz)
//  Devuelve visibilidad: 1.0 = iluminado, 0.0 = en sombra.
// ---------------------------------------------------------------------
float sombra(sampler2D mapa, vec4 pos_luz, float n_dot_l) {
    vec3 p = pos_luz.xyz / pos_luz.w * 0.5 + 0.5;   // de [-1,1] a [0,1]
    if (p.z > 1.0 || p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0)
        return 1.0;                                  // fuera del mapa: sin sombra

    float sesgo = max(0.0015 * (1.0 - n_dot_l), 0.0003);  // evita "acné" de sombra
    vec2 texel = 1.0 / vec2(textureSize(mapa, 0));
    float luz = 0.0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            float profundidad = texture(mapa, p.xy + vec2(i, j) * texel).r;
            luz += (p.z - sesgo > profundidad) ? 0.0 : 1.0;
        }
    }
    return luz / 9.0;                                // PCF: promedio de 9 muestras
}

// ---------------------------------------------------------------------
//  MODELOS DE COLOR: RGB <-> HSV
// ---------------------------------------------------------------------
vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// Rampa de riesgo: MISMOS colores y misma interpolación que rampa_riesgo() de
// vista_mapa.py (mini-mapa del HUD, Rol 1), para que HUD y escena 3D coincidan.
// verde (t=0) -> amarillo (t=0.5) -> rojo (t=1). Valores RGB 0-255 / 255.
const vec3 COLOR_VERDE    = vec3( 46.0, 204.0, 113.0) / 255.0;
const vec3 COLOR_AMARILLO = vec3(241.0, 196.0,  15.0) / 255.0;
const vec3 COLOR_ROJO     = vec3(231.0,  76.0,  60.0) / 255.0;

vec3 rampa_color(float t) {
    t = clamp(t, 0.0, 1.0);
    return (t < 0.5) ? mix(COLOR_VERDE, COLOR_AMARILLO, t / 0.5)
                     : mix(COLOR_AMARILLO, COLOR_ROJO, (t - 0.5) / 0.5);
}

// Variante en HSV (matiz 120° -> 0°): sirve para COMPARAR modelos de color en el informe.
vec3 rampa_color_hsv(float t) {
    float h = mix(120.0, 0.0, clamp(t, 0.0, 1.0)) / 360.0;
    return hsv2rgb(vec3(h, 0.85, 0.95));
}

// Rampa del agua según profundidad: somera (turquesa) -> profunda (azul oscuro)
vec3 rampa_profundidad(float d, float d_max) {
    float t = clamp(d / max(d_max, 1.0e-4), 0.0, 1.0);
    return mix(vec3(0.35, 0.75, 0.85), vec3(0.02, 0.15, 0.35), t);
}

// ---------------------------------------------------------------------
//  ESTADO DE ALARMA: oscurece y vira a rojizo cuando alarma > ~0.6
//  (el valor 'alarma' lo publica el Rol 1; aquí solo se aplica el tinte)
// ---------------------------------------------------------------------
// Suave a propósito: con 0.55 el agua quedaba casi negra y la escena no se leía.
vec3 tinte_alarma(vec3 color, float alarma) {
    vec3 oscuro  = color * 0.80;
    vec3 rojizo  = oscuro * vec3(1.25, 0.82, 0.78);
    return mix(color, rojizo, smoothstep(0.6, 1.0, alarma));
}

// ---------------------------------------------------------------------
//  NIEBLA: a lo lejos todo se funde con el color del cielo, según la
//  distancia a la cámara. Todos los shaders usan la misma, así no quedan
//  costuras entre módulos. El cielo lleva el mismo tinte de alarma que
//  el fondo que pinta main.py.
// ---------------------------------------------------------------------
float factor_niebla(vec3 pos, vec3 cam) {
    return smoothstep(650.0, 2600.0, distance(pos, cam));
}

vec3 con_niebla(vec3 color, float f, vec3 cielo, float alarma) {
    return mix(color, tinte_alarma(cielo, alarma), f);
}
