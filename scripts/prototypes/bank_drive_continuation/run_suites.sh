#!/bin/bash
# Paired suites for the drive-continuation node, one fresh process per file.
#
# The head arm runs from this worktree and the base arm from an extracted
# archive of the node's base revision, both through the shared root virtualenv
# with PYTHONPATH naming the arm's tree. Each process gets its own basetemp so
# concurrent arms cannot delete each other's, and no arm runs under a timeout.
set -u
export TMPDIR=/tmp
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
RUN=${RUN_DIR:?}
TREE=${TREE:?}
BASE=${BASE_TREE:?}
HEAD_REVISION=${HEAD_REVISION:?}
BASE_REVISION=${BASE_REVISION:?}
BASETEMPS=$(mktemp -d /tmp/"$USER"-suites-XXXX)
trap 'rm -rf "$BASETEMPS"' EXIT

arm() {
    local role=$1 tree=$2 revision=$3 file=$4
    local name
    name=$(basename "$file" .py)
    local log="$RUN/suite-$role-$name.log"
    {
        echo "# role: $role"
        echo "# revision: $revision"
        echo "# tree: $tree"
        echo "# command: PYTHONPATH=$tree $PY -m pytest -p no:cacheprovider --basetemp=$BASETEMPS/$role-$name -q $tree/tests/$name.py"
        PYTHONPATH="$tree" "$PY" -c 'import nova; print("# nova:", nova.__file__)'
        PYTHONPATH="$tree" "$PY" -m pytest -p no:cacheprovider \
            --basetemp="$BASETEMPS/$role-$name" -q "$tree/tests/$name.py"
        echo "EXIT=$?"
    } > "$log" 2>&1
    echo "SUITE $role $name $(tail -n 2 "$log" | tr '\n' ' ')"
}

ARMS=${ARMS:-head base}
if [[ " $ARMS " == *" head "* ]]; then
arm head "$TREE" "$HEAD_REVISION" "$TREE/tests/test_bank_drive_continuation.py"
arm head "$TREE" "$HEAD_REVISION" "$TREE/tests/test_bank_solve_driver.py"
arm head "$TREE" "$HEAD_REVISION" "$TREE/tests/test_bank_solve_cycle.py"
arm head "$TREE" "$HEAD_REVISION" "$TREE/tests/test_support_read.py"
fi
if [[ " $ARMS " == *" base "* ]]; then
arm base "$BASE" "$BASE_REVISION" "$BASE/tests/test_bank_solve_driver.py"
arm base "$BASE" "$BASE_REVISION" "$BASE/tests/test_bank_solve_cycle.py"
arm base "$BASE" "$BASE_REVISION" "$BASE/tests/test_support_read.py"
fi
echo "SUITES-DONE"