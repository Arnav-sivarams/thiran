if(NOT DEFINED THIRAN_EXECUTABLE OR
   NOT DEFINED SOURCE_DIR OR
   NOT DEFINED TEST_BINARY_DIR)
    message(FATAL_ERROR "RegionPlanCliTests requires executable, source, and binary paths")
endif()

set(usage
"Usage:
  Thiran <source-file>
  Thiran --plan <source-file>
  Thiran --emit-plan <source-file> <output-file>
  Thiran --emit-region-executor <source-file> <output-file>
  Thiran --version
  Thiran --help
")

file(MAKE_DIRECTORY "${TEST_BINARY_DIR}")

function(run_thiran result_var stdout_var stderr_var)
    execute_process(
        COMMAND "${THIRAN_EXECUTABLE}" ${ARGN}
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
    )
    set(${result_var} "${result}" PARENT_SCOPE)
    set(${stdout_var} "${stdout}" PARENT_SCOPE)
    set(${stderr_var} "${stderr}" PARENT_SCOPE)
endfunction()

function(require_equal actual expected label)
    if(NOT "${actual}" STREQUAL "${expected}")
        message(FATAL_ERROR "${label}: values differ")
    endif()
endfunction()

function(require_plan output label)
    if(NOT "${output}" MATCHES "^HybridRegionPlan")
        message(FATAL_ERROR "${label}: missing HybridRegionPlan prefix")
    endif()
    foreach(token "Nodes:" "Regions:" "Dependencies:" "Decisions")
        string(FIND "${output}" "${token}" position)
        if(position EQUAL -1)
            message(FATAL_ERROR "${label}: missing ${token}")
        endif()
    endforeach()
    if(NOT "${output}" MATCHES "(AOT|JIT|FALLBACK)")
        message(FATAL_ERROR "${label}: missing strategy token")
    endif()
    string(FIND "${output}" "0x" pointer_position)
    if(NOT pointer_position EQUAL -1)
        message(FATAL_ERROR "${label}: contains pointer address")
    endif()
    string(LENGTH "${output}" output_length)
    if(output_length EQUAL 0)
        message(FATAL_ERROR "${label}: empty output")
    endif()
    math(EXPR last_index "${output_length} - 1")
    string(SUBSTRING "${output}" "${last_index}" 1 final_character)
    if(NOT final_character STREQUAL "\n")
        message(FATAL_ERROR "${label}: missing final newline")
    endif()
endfunction()

# Cases 1a and 1b: help.
run_thiran(result stdout stderr --help)
require_equal("${result}" "0" "help exit")
require_equal("${stdout}" "${usage}" "help stdout")
require_equal("${stderr}" "" "help stderr")
run_thiran(result stdout stderr -h)
require_equal("${result}" "0" "short help exit")
require_equal("${stdout}" "${usage}" "short help stdout")
require_equal("${stderr}" "" "short help stderr")

# Case 2: invalid command.
run_thiran(result stdout stderr --unknown)
require_equal("${result}" "2" "invalid command exit")
require_equal("${stdout}" "" "invalid command stdout")
require_equal("${stderr}" "Invalid command line\n${usage}" "invalid command stderr")

# Case 3: missing plan source.
run_thiran(result stdout stderr --plan)
require_equal("${result}" "2" "missing plan source exit")
require_equal("${stdout}" "" "missing plan source stdout")
string(FIND "${stderr}" "${usage}" usage_position)
if(usage_position EQUAL -1)
    message(FATAL_ERROR "missing plan source: usage absent")
endif()

# Cases 4 and 5: plan CNN and MLP.
run_thiran(cnn_result cnn_plan cnn_error --plan examples/cnn.th)
require_equal("${cnn_result}" "0" "CNN plan exit")
require_equal("${cnn_error}" "" "CNN plan stderr")
require_plan("${cnn_plan}" "CNN plan")
run_thiran(mlp_result mlp_plan mlp_error --plan examples/mlp.th)
require_equal("${mlp_result}" "0" "MLP plan exit")
require_equal("${mlp_error}" "" "MLP plan stderr")
require_plan("${mlp_plan}" "MLP plan")

# Case 6: plan-mode artifact isolation.
foreach(artifact generated.py graph.dot)
    if(EXISTS "${SOURCE_DIR}/${artifact}")
        set("${artifact}_existed" TRUE)
        file(READ "${SOURCE_DIR}/${artifact}" "${artifact}_before")
    else()
        set("${artifact}_existed" FALSE)
    endif()
endforeach()
run_thiran(result stdout stderr --plan examples/cnn.th)
run_thiran(result stdout stderr --plan examples/mlp.th)
foreach(artifact generated.py graph.dot)
    if(${artifact}_existed)
        if(NOT EXISTS "${SOURCE_DIR}/${artifact}")
            message(FATAL_ERROR "${artifact} was removed by plan mode")
        endif()
        file(READ "${SOURCE_DIR}/${artifact}" artifact_after)
        if(NOT "${artifact_after}" STREQUAL "${${artifact}_before}")
            message(FATAL_ERROR "${artifact} changed in plan mode")
        endif()
    elseif(EXISTS "${SOURCE_DIR}/${artifact}")
        message(FATAL_ERROR "${artifact} was created by plan mode")
    endif()
endforeach()
if(EXISTS "${SOURCE_DIR}/region_plan.txt")
    message(FATAL_ERROR "plan mode created region_plan.txt")
endif()

# Cases 7 and 8: emit plan and print/emit equivalence.
set(plan_file "${TEST_BINARY_DIR}/region-plan.txt")
file(REMOVE "${plan_file}")
run_thiran(result stdout stderr
    --emit-plan examples/cnn.th "${plan_file}")
require_equal("${result}" "0" "emit exit")
require_equal(
    "${stdout}"
    "Wrote hybrid region plan: ${plan_file}\n"
    "emit stdout"
)
require_equal("${stderr}" "" "emit stderr")
if(NOT EXISTS "${plan_file}")
    message(FATAL_ERROR "emit output does not exist")
endif()
file(READ "${plan_file}" emitted_plan)
require_plan("${emitted_plan}" "emitted plan")
require_equal("${emitted_plan}" "${cnn_plan}" "print and emit equivalence")

# Case 9: overwrite output.
file(WRITE "${plan_file}" "old content")
run_thiran(result stdout stderr
    --emit-plan examples/cnn.th "${plan_file}")
require_equal("${result}" "0" "overwrite exit")
file(READ "${plan_file}" overwritten_plan)
require_equal("${overwritten_plan}" "${cnn_plan}" "overwrite contents")

# Case 10: missing parent.
set(missing_parent "${TEST_BINARY_DIR}/missing-parent")
file(REMOVE_RECURSE "${missing_parent}")
set(missing_output "${missing_parent}/plan.txt")
run_thiran(result stdout stderr
    --emit-plan examples/cnn.th "${missing_output}")
require_equal("${result}" "4" "missing parent exit")
require_equal("${stdout}" "" "missing parent stdout")
require_equal(
    "${stderr}"
    "Failed to open RegionPlan output: ${missing_output}\n"
    "missing parent stderr"
)
if(EXISTS "${missing_parent}")
    message(FATAL_ERROR "missing parent directory was created")
endif()

# Case 11: source equals output.
set(source_copy "${TEST_BINARY_DIR}/source-copy.th")
file(COPY_FILE "${SOURCE_DIR}/examples/cnn.th" "${source_copy}")
file(READ "${source_copy}" source_before)
run_thiran(result stdout stderr
    --emit-plan "${source_copy}" "${source_copy}")
require_equal("${result}" "4" "source equals output exit")
require_equal("${stdout}" "" "source equals output stdout")
require_equal(
    "${stderr}"
    "Refusing to overwrite source file with RegionPlan: ${source_copy}\n"
    "source equals output stderr"
)
file(READ "${source_copy}" source_after)
require_equal("${source_after}" "${source_before}" "source preservation")

# Case 12: extra argument.
run_thiran(result stdout stderr --plan examples/cnn.th extra)
require_equal("${result}" "2" "extra argument exit")
require_equal("${stdout}" "" "extra argument stdout")
string(FIND "${stderr}" "${usage}" usage_position)
if(usage_position EQUAL -1)
    message(FATAL_ERROR "extra argument: usage absent")
endif()
