#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# CONFIGURACIÓN DE LOS EXPERIMENTOS
# ============================================================
PROCESS_COUNTS=(3 6 9 12 15 18 21 24)
REPETITIONS=1
PROBLEM_SIZES=(5000 10000 20000)

# ============================================================
# CONFIGURACIÓN DEL CLÚSTER
# ============================================================
CLUSTER=0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${HOSTFILE:-$ROOT/hostfile}"

BUILD_DIR="${BUILD_DIR:-build-opt}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SEQUENTIAL_TARGET="${SEQUENTIAL_TARGET:-cv}"
STATIC_SOURCE="${STATIC_SOURCE:-main_mpi_estatico.cc}"
DYNAMIC_SOURCE="${DYNAMIC_SOURCE:-main_mpi_dinamico.cc}"
STATIC_BIN="${STATIC_BIN:-cv_mpi_estatico}"
DYNAMIC_BIN="${DYNAMIC_BIN:-cv_mpi_dinamico}"

VERSION="secuencial"
DOWNLOAD=0
RUN_TESTS=1

usage() {
  cat <<'EOF'
Uso: ./script.sh [--secuencial | --estatica | --dinamica] [--descargar] [--sin-tests] [--cluster] [--hostfile FILE]
  --secuencial  Ejecuta la versión secuencial.
  --estatica    Ejecuta MPI estática.
  --dinamica    Ejecuta MPI dinámica.
  --descargar   Ejecuta get_dataset.sh antes de los experimentos.
  --sin-tests   Omite CMake y ctest.
  --cluster     Distribuye el proyecto por SSH y ejecuta MPI en el clúster.
  --hostfile    Selecciona un hostfile (por defecto ./hostfile).
  -h, --help    Muestra esta ayuda.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --secuencial) VERSION="secuencial" ;;
    --estatica) VERSION="estatica" ;;
    --dinamica) VERSION="dinamica" ;;
    --descargar) DOWNLOAD=1 ;;
    --sin-tests) RUN_TESTS=0 ;;
    --cluster) CLUSTER=1 ;;
    --hostfile)
      shift
      [[ $# -gt 0 ]] || { echo "ERROR: falta el archivo hostfile." >&2; exit 2; }
      HOSTFILE="$1"
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Opción desconocida: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: REPETITIONS debe ser positivo." >&2; exit 2; }
((${#PROBLEM_SIZES[@]} > 0)) || { echo "ERROR: PROBLEM_SIZES está vacío." >&2; exit 2; }
for size in "${PROBLEM_SIZES[@]}"; do
  [[ "$size" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: tamaño no válido: $size" >&2; exit 2; }
done
if [[ "$VERSION" != "secuencial" ]]; then
  ((${#PROCESS_COUNTS[@]} > 0)) || { echo "ERROR: PROCESS_COUNTS está vacío." >&2; exit 2; }
  for n in "${PROCESS_COUNTS[@]}"; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: número de procesos no válido: $n" >&2; exit 2; }
  done
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
HOSTFILE="${HOSTFILE:-$ROOT/hostfile}"
[[ -f "$ROOT/CMakeLists.txt" ]] || { echo "ERROR: no encuentro CMakeLists.txt en $ROOT." >&2; exit 1; }
if [[ "$CLUSTER" -eq 1 && "$VERSION" == "secuencial" ]]; then
  echo "ERROR: --cluster solo puede usarse con --estatica o --dinamica." >&2; exit 2
fi

# ============================================================
# DATOS MNIST
# ============================================================
if [[ "$DOWNLOAD" -eq 1 ]]; then
  [[ -f "$ROOT/get_dataset.sh" ]] || { echo "ERROR: no encuentro get_dataset.sh." >&2; exit 1; }
  echo "==> Descargando/preparando MNIST..."
  bash "$ROOT/get_dataset.sh"
fi
DATA_FILES=(
  "$ROOT/build/train/train-images-idx3-ubyte"
  "$ROOT/build/train/train-labels-idx1-ubyte"
  "$ROOT/build/test/t10k-images-idx3-ubyte"
  "$ROOT/build/test/t10k-labels-idx1-ubyte"
)
for file in "${DATA_FILES[@]}"; do
  [[ -s "$file" ]] || { echo "ERROR: falta MNIST: $file" >&2; echo "Ejecuta: $0 --descargar" >&2; exit 1; }
done

# ============================================================
# COMPILACIÓN Y TESTS
# ============================================================
if [[ "$RUN_TESTS" -eq 1 ]]; then
  echo "==> Configurando compilación Release..."
  cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG"
  echo "==> Compilando objetivos CMake..."
  cmake --build "$BUILD_DIR" --parallel "$(nproc)"
  echo "==> Ejecutando tests..."
  ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

# Enlaces de datos esperados por los programas desde BUILD_DIR.
mkdir -p "$BUILD_DIR/train" "$BUILD_DIR/test"
ln -sfn "$ROOT/build/train/train-images-idx3-ubyte" "$BUILD_DIR/train/train-images-idx3-ubyte"
ln -sfn "$ROOT/build/train/train-labels-idx1-ubyte" "$BUILD_DIR/train/train-labels-idx1-ubyte"
ln -sfn "$ROOT/build/test/t10k-images-idx3-ubyte" "$BUILD_DIR/test/t10k-images-idx3-ubyte"
ln -sfn "$ROOT/build/test/t10k-labels-idx1-ubyte" "$BUILD_DIR/test/t10k-labels-idx1-ubyte"

if [[ "$VERSION" == "estatica" ]]; then
  [[ -f "$STATIC_SOURCE" ]] || { echo "ERROR: no encuentro $STATIC_SOURCE." >&2; exit 1; }
  command -v mpic++ >/dev/null || { echo "ERROR: mpic++ no está instalado." >&2; exit 1; }
  echo "==> Compilando versión MPI estática..."
  mpic++ -std=c++17 -O3 -DNDEBUG "$STATIC_SOURCE" crossValidation.cc mnist.cc knn.cc sample.cc -o "$BUILD_DIR/$STATIC_BIN"
elif [[ "$VERSION" == "dinamica" ]]; then
  [[ -f "$DYNAMIC_SOURCE" ]] || { echo "ERROR: no encuentro $DYNAMIC_SOURCE." >&2; exit 1; }
  command -v mpic++ >/dev/null || { echo "ERROR: mpic++ no está instalado." >&2; exit 1; }
  echo "==> Compilando versión MPI dinámica..."
  mpic++ -std=c++17 -O3 -DNDEBUG "$DYNAMIC_SOURCE" crossValidation.cc mnist.cc knn.cc sample.cc -o "$BUILD_DIR/$DYNAMIC_BIN"
else
  [[ -x "$BUILD_DIR/$SEQUENTIAL_TARGET" ]] || { echo "ERROR: no encuentro $BUILD_DIR/$SEQUENTIAL_TARGET." >&2; exit 1; }
fi

# ============================================================
# DISTRIBUCIÓN POR SSH
# ============================================================
sync_cluster() {
  [[ -f "$HOSTFILE" ]] || { echo "ERROR: no existe el hostfile: $HOSTFILE" >&2; exit 1; }
  command -v ssh >/dev/null || { echo "ERROR: ssh no está instalado." >&2; exit 1; }
  command -v rsync >/dev/null || { echo "ERROR: rsync no está instalado." >&2; exit 1; }
  echo "==> Distribuyendo proyecto por SSH..."
  local host rest
  while read -r host rest || [[ -n "${host:-}" ]]; do
    [[ -z "${host:-}" || "$host" == \#* ]] && continue
    # El hostfile usa host [slots=N]. El primer campo identifica el equipo.
    [[ "$host" == "localhost" || "$host" == "$(hostname)" || "$host" == "$(hostname -s)" ]] && continue
    echo "==> Comprobando conexión SSH con $host..."
    ssh -o BatchMode=yes "$host" "mkdir -p '$ROOT'"
    echo "==> Copiando archivos a $host:$ROOT"
    rsync -az --delete --exclude='.git/' --exclude='results/' "$ROOT/" "$host:$ROOT/"
  done < "$HOSTFILE"
  echo "==> Distribución completada."
}
if [[ "$CLUSTER" -eq 1 ]]; then sync_cluster; fi

# ============================================================
# CONFIGURACIÓN DE RESULTADOS
# ============================================================
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
OUT_DIR="$ROOT/$RESULTS_DIR/$VERSION"
mkdir -p "$OUT_DIR"
RESULT_FILE="$OUT_DIR/${VERSION}_experimentos_${TIMESTAMP}.txt"
if [[ "$VERSION" == "secuencial" ]]; then
  MPI_COUNTS=(1); EXECUTABLE="$ROOT/$BUILD_DIR/$SEQUENTIAL_TARGET"
elif [[ "$VERSION" == "estatica" ]]; then
  MPI_COUNTS=("${PROCESS_COUNTS[@]}"); EXECUTABLE="$ROOT/$BUILD_DIR/$STATIC_BIN"
else
  MPI_COUNTS=("${PROCESS_COUNTS[@]}"); EXECUTABLE="$ROOT/$BUILD_DIR/$DYNAMIC_BIN"
fi
{
  echo "EXPERIMENTOS KNN - MNIST"
  echo "Fecha de inicio: $(date)"
  echo "Versión: $VERSION"
  echo "Modo clúster: $CLUSTER"
  echo "Hostfile: $HOSTFILE"
  echo "Tamaños: ${PROBLEM_SIZES[*]}"
  echo "Procesos: ${MPI_COUNTS[*]}"
  echo "Repeticiones: $REPETITIONS"
  echo "Equipo principal: $(hostname)"
  echo "============================================================"
} | tee "$RESULT_FILE"

print_stats() {
  local -n values_ref=$1
  awk -v values="${values_ref[*]}" 'BEGIN {
    n=split(values,a," "); sum=0; min=a[1]; max=a[1]
    for(i=1;i<=n;i++){x=a[i]+0;sum+=x;if(x<min)min=x;if(x>max)max=x}
    mean=sum/n; sq=0
    for(i=1;i<=n;i++){d=(a[i]+0)-mean;sq+=d*d}
    printf "Repeticiones válidas: %d\nMedia: %.3f s\nDesviación típica: %.3f s\nMínimo: %.3f s\nMáximo: %.3f s\n",n,mean,sqrt(sq/n),min,max
  }'
}

# ============================================================
# EJECUCIÓN DE LOS EXPERIMENTOS
# ============================================================
for problem_size in "${PROBLEM_SIZES[@]}"; do
  for np in "${MPI_COUNTS[@]}"; do
    TIMES=()
    { echo; echo "############################################################"; echo "RESUMEN: tamaño=$problem_size, procesos=$np"; echo "############################################################"; } | tee -a "$RESULT_FILE"
    for ((rep=1; rep<=REPETITIONS; rep++)); do
      RUN_TMP="$(mktemp)"; TIME_TMP="$(mktemp)"
      echo "==> $VERSION | tamaño=$problem_size | procesos=$np | repetición=$rep/$REPETITIONS"
      { echo; echo "EJECUCIÓN $rep/$REPETITIONS - $(date)"; echo "Tamaño: $problem_size | Procesos: $np | Versión: $VERSION"; } | tee -a "$RESULT_FILE"
      set +e
      if [[ "$VERSION" == "secuencial" ]]; then
        (cd "$ROOT/$BUILD_DIR"; env KNN_TRAIN_SIZE="$problem_size" /usr/bin/time -f 'Tiempo real total: %e s\nCPU: %P\nMemoria máxima: %M KB' -o "$TIME_TMP" "./$SEQUENTIAL_TARGET") >"$RUN_TMP" 2>&1
      elif [[ "$CLUSTER" -eq 1 ]]; then
        (cd "$ROOT/$BUILD_DIR"; env KNN_TRAIN_SIZE="$problem_size" /usr/bin/time -f 'Tiempo real total: %e s\nCPU: %P\nMemoria máxima: %M KB' -o "$TIME_TMP" mpirun --hostfile "$HOSTFILE" --wdir "$ROOT/$BUILD_DIR" -np "$np" -x KNN_TRAIN_SIZE "$EXECUTABLE") >"$RUN_TMP" 2>&1
      else
        (cd "$ROOT/$BUILD_DIR"; env KNN_TRAIN_SIZE="$problem_size" /usr/bin/time -f 'Tiempo real total: %e s\nCPU: %P\nMemoria máxima: %M KB' -o "$TIME_TMP" mpirun -np "$np" -x KNN_TRAIN_SIZE "$EXECUTABLE") >"$RUN_TMP" 2>&1
      fi
      STATUS=$?
      set -e
      cat "$RUN_TMP" | tee -a "$RESULT_FILE"
      [[ ! -s "$TIME_TMP" ]] || cat "$TIME_TMP" | tee -a "$RESULT_FILE"
      REAL_TIME="$(awk '/Tiempo real total:/ {print $(NF-1)}' "$TIME_TMP" | tail -n 1)"
      if [[ "$STATUS" -eq 0 && "$REAL_TIME" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        TIMES+=("$REAL_TIME")
        echo "Resultado: CORRECTO | tiempo real: $REAL_TIME s" | tee -a "$RESULT_FILE"
      else
        echo "ERROR: ejecución fallida (código $STATUS) o tiempo no disponible." | tee -a "$RESULT_FILE"
      fi
      rm -f "$RUN_TMP" "$TIME_TMP"
    done
    { echo; echo "ESTADÍSTICAS - tamaño=$problem_size, procesos=$np"; if ((${#TIMES[@]} > 0)); then print_stats TIMES; else echo "No hay ejecuciones válidas."; fi; echo "============================================================"; } | tee -a "$RESULT_FILE"
  done
done
{
  echo; echo "FIN DE LOS EXPERIMENTOS: $(date)"; echo "Fichero combinado: $RESULT_FILE"
} | tee -a "$RESULT_FILE"
echo "==> Experimentos terminados. Resultados: $RESULT_FILE"
