if(NOT DEFINED THIRAN_EXECUTABLE OR NOT DEFINED SOURCE_DIR OR
   NOT DEFINED TEST_BINARY_DIR OR NOT DEFINED PYTHON_EXECUTABLE OR
   NOT DEFINED RUNTIME_TEST_EXECUTABLE)
    message(FATAL_ERROR "RegionRuntimeCliTests requires executable, source, binary, and Python paths")
endif()

file(REMOVE_RECURSE "${TEST_BINARY_DIR}")
file(MAKE_DIRECTORY "${TEST_BINARY_DIR}")
set(cnn "${SOURCE_DIR}/examples/cnn.th")
set(mlp "${SOURCE_DIR}/examples/mlp.th")
set(fixture "${SOURCE_DIR}/tests/fixtures/region_execution_multi_output_constant.th")
set(dynamic_fixture "${SOURCE_DIR}/tests/fixtures/region_runtime_dynamic_input.th")
set(usage "Usage:
  ThiranRegionExecutor --describe
  ThiranRegionExecutor --run <input-bundle.pt> <output-bundle.pt>
  ThiranRegionExecutor --help
")

function(run result_var stdout_var stderr_var)
    execute_process(
        COMMAND ${ARGN}
        WORKING_DIRECTORY "${TEST_BINARY_DIR}"
        RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    set(${result_var} "${result}" PARENT_SCOPE)
    set(${stdout_var} "${stdout}" PARENT_SCOPE)
    set(${stderr_var} "${stderr}" PARENT_SCOPE)
endfunction()
function(equal actual expected label)
    if(NOT "${actual}" STREQUAL "${expected}")
        message(FATAL_ERROR "${label}: expected=[${expected}] actual=[${actual}]")
    endif()
endfunction()
function(contains text token label)
    string(FIND "${text}" "${token}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "${label}: missing [${token}]")
    endif()
endfunction()
function(emit source output)
    run(result stdout stderr "${THIRAN_EXECUTABLE}"
        --emit-region-executor "${source}" "${output}")
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "executor emission failed: ${stderr}")
    endif()
endfunction()

set(cnn_py "${TEST_BINARY_DIR}/cnn.py")
set(mlp_py "${TEST_BINARY_DIR}/mlp.py")
set(fixture_py "${TEST_BINARY_DIR}/fixture.py")
set(dynamic_py "${TEST_BINARY_DIR}/dynamic.py")
set(empty_py "${TEST_BINARY_DIR}/empty.py")
emit("${cnn}" "${cnn_py}")
emit("${mlp}" "${mlp_py}")
emit("${fixture}" "${fixture_py}")
emit("${dynamic_fixture}" "${dynamic_py}")
run(result stdout stderr "${RUNTIME_TEST_EXECUTABLE}" --emit-empty "${empty_py}")
equal("${result}" "0" "empty artifact emission")

# Cases 1-2: help and invalid usage.
foreach(option --help -h)
    run(result stdout stderr "${PYTHON_EXECUTABLE}" "${cnn_py}" "${option}")
    equal("${result}" "0" "help exit")
    equal("${stdout}" "${usage}" "help stdout")
    equal("${stderr}" "" "help stderr")
endforeach()
foreach(arguments "" "--unknown" "--describe;extra" "--run"
                  "--run;input.pt" "--run;input.pt;output.pt;extra"
                  "--help;extra")
    run(result stdout stderr "${PYTHON_EXECUTABLE}" "${cnn_py}" ${arguments})
    equal("${result}" "2" "invalid usage exit")
    equal("${stdout}" "" "invalid usage stdout")
    equal("${stderr}" "Invalid Region executor command\n${usage}" "invalid usage stderr")
endforeach()

# Cases 3-4: exact descriptions.
run(result stdout stderr "${PYTHON_EXECUTABLE}" "${cnn_py}" --describe)
equal("${result}" "0" "CNN describe exit")
equal("${stderr}" "" "CNN describe stderr")
equal("${stdout}" "Thiran Region Executor
Inputs:
  'Image' shape=(1, 1, 8, 8)
  'Filter' shape=(2, 1, 3, 3)
  'Classifier' shape=(18, 4)
Outputs:
  'O'
" "CNN description")
run(result stdout stderr "${PYTHON_EXECUTABLE}" "${mlp_py}" --describe)
equal("${stdout}" "Thiran Region Executor
Inputs:
  'X' shape=(2, 8)
  'W1' shape=(8, 16)
  'B1' shape=(2, 16)
  'W2' shape=(16, 4)
  'B2' shape=(2, 4)
Outputs:
  'O'
" "MLP description")
run(result stdout stderr "${PYTHON_EXECUTABLE}" "${empty_py}" --describe)
equal("${result}" "0" "empty describe exit")
equal("${stderr}" "" "empty describe stderr")
equal("${stdout}" "Thiran Region Executor
Inputs:
  <none>
Outputs:
  <none>
" "empty description")

# Create deterministic valid and malformed bundles.
set(make_bundles "${TEST_BINARY_DIR}/make_bundles.py")
file(WRITE "${make_bundles}" "import torch\n"
"torch.save({'Image': torch.arange(64,dtype=torch.float32).reshape(1,1,8,8)/64,"
"'Filter': torch.full((2,1,3,3),.125),"
"'Classifier': torch.arange(72,dtype=torch.float32).reshape(18,4)/72},r'${TEST_BINARY_DIR}/cnn-input.pt')\n"
"torch.save({'X': torch.arange(16,dtype=torch.float32).reshape(2,8)/16,"
"'W1': torch.arange(128,dtype=torch.float32).reshape(8,16)/128,"
"'B1': torch.full((2,16),.25),'W2': torch.arange(64,dtype=torch.float32).reshape(16,4)/64,"
"'B2': torch.full((2,4),-.5)},r'${TEST_BINARY_DIR}/mlp-input.pt')\n"
"torch.save({'X': torch.arange(4,dtype=torch.float32).reshape(2,2)},r'${TEST_BINARY_DIR}/fixture-input.pt')\n"
"torch.save({'X': torch.arange(8,dtype=torch.float32).reshape(2,4)},r'${TEST_BINARY_DIR}/dynamic-two.pt')\n"
"torch.save({'X': torch.arange(20,dtype=torch.float32).reshape(5,4)},r'${TEST_BINARY_DIR}/dynamic-five.pt')\n"
"torch.save({'X': torch.arange(10,dtype=torch.float32).reshape(2,5)},r'${TEST_BINARY_DIR}/dynamic-bad.pt')\n"
"torch.save(torch.tensor(1),r'${TEST_BINARY_DIR}/not-dict.pt')\n"
"torch.save({1: torch.tensor(1)},r'${TEST_BINARY_DIR}/bad-key.pt')\n"
"torch.save({'Image':'bad','Filter':torch.zeros(2,1,3,3),'Classifier':torch.zeros(18,4)},r'${TEST_BINARY_DIR}/bad-value.pt')\n"
"torch.save({'Image':torch.zeros(1,1,8,8)},r'${TEST_BINARY_DIR}/missing.pt')\n"
"torch.save({'Image':torch.zeros(1,1,8,8),'Filter':torch.zeros(2,1,3,3),'Classifier':torch.zeros(18,4),'extra':torch.tensor(1)},r'${TEST_BINARY_DIR}/extra.pt')\n"
"torch.save({'Image':torch.zeros(8,8),'Filter':torch.zeros(2,1,3,3),'Classifier':torch.zeros(18,4)},r'${TEST_BINARY_DIR}/rank.pt')\n"
"torch.save({'Image':torch.zeros(1,1,7,8),'Filter':torch.zeros(2,1,3,3),'Classifier':torch.zeros(18,4)},r'${TEST_BINARY_DIR}/dimension.pt')\n")
run(result stdout stderr "${PYTHON_EXECUTABLE}" "${make_bundles}")
equal("${result}" "0" "bundle creation")

function(run_valid label artifact input output expected)
    run(result stdout stderr "${PYTHON_EXECUTABLE}" "${artifact}" --run "${input}" "${output}")
    equal("${result}" "0" "${label} exit")
    equal("${stdout}" "Wrote Region outputs: ${output}\n" "${label} stdout")
    equal("${stderr}" "" "${label} stderr")
    run(check_result check_stdout check_stderr "${PYTHON_EXECUTABLE}" -c
        "import torch; o=torch.load(r'${output}',map_location='cpu',weights_only=True); assert list(o)==${expected}; assert all(v.device.type=='cpu' for v in o.values())")
    equal("${check_result}" "0" "${label} safe output")
endfunction()

# Cases 6-9: execution and output bundle contracts.
run_valid("CNN" "${cnn_py}" "${TEST_BINARY_DIR}/cnn-input.pt" "${TEST_BINARY_DIR}/cnn-output.pt" "['O']")
run_valid("MLP" "${mlp_py}" "${TEST_BINARY_DIR}/mlp-input.pt" "${TEST_BINARY_DIR}/mlp-output.pt" "['O']")
run_valid("fixture" "${fixture_py}" "${TEST_BINARY_DIR}/fixture-input.pt" "${TEST_BINARY_DIR}/fixture-output.pt" "['O1','O2']")
run_valid("dynamic two" "${dynamic_py}" "${TEST_BINARY_DIR}/dynamic-two.pt" "${TEST_BINARY_DIR}/dynamic-two-output.pt" "['O']")
run_valid("dynamic five" "${dynamic_py}" "${TEST_BINARY_DIR}/dynamic-five.pt" "${TEST_BINARY_DIR}/dynamic-five-output.pt" "['O']")
run(result stdout stderr "${PYTHON_EXECUTABLE}" "${dynamic_py}" --run
    "${TEST_BINARY_DIR}/dynamic-bad.pt" "${TEST_BINARY_DIR}/dynamic-bad-output.pt")
equal("${result}" "3" "dynamic fixed dimension exit")
equal("${stdout}" "" "dynamic fixed dimension stdout")
equal("${stderr}" "Invalid input tensor bundle: input 'X' dimension 1 expected 4 but found 5\n" "dynamic fixed dimension stderr")

# Cases 10-18: controlled input failures.
file(WRITE "${TEST_BINARY_DIR}/corrupt.pt" "not a tensor bundle")
foreach(case
    "missing-file.pt|Failed to load input tensor bundle: ${TEST_BINARY_DIR}/missing-file.pt"
    "corrupt.pt|Failed to load input tensor bundle: ${TEST_BINARY_DIR}/corrupt.pt"
    "not-dict.pt|Invalid input tensor bundle: input bundle is not a dictionary"
    "bad-key.pt|Invalid input tensor bundle: input bundle contains a non-string key"
    "bad-value.pt|Invalid input tensor bundle: input 'Image' is not a torch.Tensor"
    "missing.pt|Invalid input tensor bundle: missing required input 'Filter'"
    "extra.pt|Invalid input tensor bundle: unexpected input 'extra'"
    "rank.pt|Invalid input tensor bundle: input 'Image' expected rank 4 but found 2"
    "dimension.pt|Invalid input tensor bundle: input 'Image' dimension 2 expected 8 but found 7")
    string(REPLACE "|" ";" parts "${case}")
    list(GET parts 0 file)
    list(GET parts 1 message)
    set(output "${TEST_BINARY_DIR}/failure-output.pt")
    file(REMOVE "${output}")
    run(result stdout stderr "${PYTHON_EXECUTABLE}" "${cnn_py}" --run
        "${TEST_BINARY_DIR}/${file}" "${output}")
    equal("${result}" "3" "${file} exit")
    equal("${stdout}" "" "${file} stdout")
    equal("${stderr}" "${message}\n" "${file} stderr")
    if(EXISTS "${output}")
        message(FATAL_ERROR "${file} created output")
    endif()
endforeach()

# Cases 20-22: path safety and overwrite.
file(SHA256 "${TEST_BINARY_DIR}/cnn-input.pt" before_hash)
run(result stdout stderr "${PYTHON_EXECUTABLE}" "${cnn_py}" --run
    "${TEST_BINARY_DIR}/cnn-input.pt" "${TEST_BINARY_DIR}/cnn-input.pt")
equal("${result}" "5" "same path exit")
file(SHA256 "${TEST_BINARY_DIR}/cnn-input.pt" after_hash)
equal("${before_hash}" "${after_hash}" "same path preservation")
set(missing_parent "${TEST_BINARY_DIR}/absent")
file(REMOVE_RECURSE "${missing_parent}")
run(result stdout stderr "${PYTHON_EXECUTABLE}" "${cnn_py}" --run
    "${TEST_BINARY_DIR}/cnn-input.pt" "${missing_parent}/out.pt")
equal("${result}" "5" "missing parent exit")
if(EXISTS "${missing_parent}")
    message(FATAL_ERROR "parent created")
endif()
file(WRITE "${TEST_BINARY_DIR}/overwrite.pt" "old")
run_valid("overwrite" "${cnn_py}" "${TEST_BINARY_DIR}/cnn-input.pt"
    "${TEST_BINARY_DIR}/overwrite.pt" "['O']")

# Forced serialization failure after successful execution.
set(atomic_driver "${TEST_BINARY_DIR}/atomic-failure.py")
file(WRITE "${atomic_driver}"
"import contextlib, importlib.util, io, pathlib\n"
"spec=importlib.util.spec_from_file_location('runtime',r'${cnn_py}')\n"
"module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
"output=pathlib.Path(r'${TEST_BINARY_DIR}/atomic-output.pt'); output.write_bytes(b'preserved')\n"
"before=set(output.parent.iterdir()); original=module.torch.save\n"
"def fail(*args,**kwargs): raise RuntimeError('forced serialization failure')\n"
"module.torch.save=fail; stdout=io.StringIO(); stderr=io.StringIO()\n"
"try:\n"
" with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):\n"
"  code=module.main(['--run',r'${TEST_BINARY_DIR}/cnn-input.pt',str(output)])\n"
"finally: module.torch.save=original\n"
"assert code==5 and stdout.getvalue()==''\n"
"assert stderr.getvalue()==f'Failed to write output tensor bundle: {output}\\n'\n"
"assert output.read_bytes()==b'preserved' and set(output.parent.iterdir())==before\n")
run(result stdout stderr "${PYTHON_EXECUTABLE}" "${atomic_driver}")
equal("${result}" "0" "forced serialization atomicity")

function(compare_whole label source input output)
    run(result stdout stderr "${THIRAN_EXECUTABLE}" "${source}")
    equal("${result}" "0" "${label} whole compile")
    set(script "${TEST_BINARY_DIR}/${label}-whole.py")
    file(WRITE "${script}"
"import importlib.util,torch\n"
"s=importlib.util.spec_from_file_location('whole',r'${TEST_BINARY_DIR}/generated.py')\n"
"m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
"inputs=torch.load(r'${input}',map_location='cpu',weights_only=True)\n"
"expected=m.run_compiled_graph(inputs)\n"
"actual=torch.load(r'${output}',map_location='cpu',weights_only=True)\n"
"assert list(actual)==list(expected)\n"
"for key in expected:\n"
" assert actual[key].shape==expected[key].shape and actual[key].dtype==expected[key].dtype\n"
" assert actual[key].device.type=='cpu'\n"
" torch.testing.assert_close(actual[key],expected[key].cpu(),rtol=1e-5,atol=1e-6)\n")
    run(result stdout stderr "${CMAKE_COMMAND}" -E env "CUDA_VISIBLE_DEVICES="
        "${PYTHON_EXECUTABLE}" "${script}")
    equal("${result}" "0" "${label} whole equivalence")
endfunction()
compare_whole("cnn" "${cnn}" "${TEST_BINARY_DIR}/cnn-input.pt" "${TEST_BINARY_DIR}/cnn-output.pt")
compare_whole("mlp" "${mlp}" "${TEST_BINARY_DIR}/mlp-input.pt" "${TEST_BINARY_DIR}/mlp-output.pt")
compare_whole("fixture" "${fixture}" "${TEST_BINARY_DIR}/fixture-input.pt" "${TEST_BINARY_DIR}/fixture-output.pt")

# Cases 24-25: import safety and library extra-key compatibility.
run(result stdout stderr "${PYTHON_EXECUTABLE}" -c
    "import importlib.util,torch; p=r'${cnn_py}'; s=importlib.util.spec_from_file_location('r',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert callable(m.run_region_plan); assert m.thiran_output_names()==('O',); i=torch.load(r'${TEST_BINARY_DIR}/cnn-input.pt',weights_only=True); i['extra']=torch.tensor(1); assert list(m.run_region_plan(i))==['O']")
equal("${result}" "0" "import and extra-key compatibility")
equal("${stdout}" "" "import stdout")
equal("${stderr}" "" "import stderr")

# Security and isolation checks.
file(READ "${cnn_py}" generated)
contains("${generated}" "weights_only=True" "restricted loading")
foreach(forbidden "import pickle" "eval(" "exec(" "subprocess" "import triton")
    string(FIND "${generated}" "${forbidden}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "forbidden source token ${forbidden}")
    endif()
endforeach()
