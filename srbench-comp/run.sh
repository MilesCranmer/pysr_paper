TASKVERSION=$1
TASKS=$2
# Set LOCAL with default value of 0:
LOCAL=${3:-0}
# Check if run local or not:
if [[ $LOCAL -eq "1" ]]; then
    ./generate_tasks.sh $1 > "tasks_v${TASKVERSION}.sh" && chmod +x "tasks_v${TASKVERSION}.sh"
    disBatch -s localhost:$TASKS -g "tasks_v${TASKVERSION}.sh" -p logs
else
    ./generate_tasks.sh $1 > "tasks_v${TASKVERSION}.sh" && chmod +x "tasks_v${TASKVERSION}.sh"
    sbatch -n $TASKS -c 8 -p cca --constraint='rome,ib' --reservation=rocky8 disBatch -g "tasks_v${TASKVERSION}.sh" -p logs
fi