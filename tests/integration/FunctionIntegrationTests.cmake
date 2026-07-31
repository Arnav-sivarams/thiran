cmake_minimum_required(VERSION 3.20)
if(NOT THIRAN_EXECUTABLE OR NOT SOURCE_DIR OR NOT TEST_BINARY_DIR OR NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "FunctionIntegrationTests requires executable, source, binary, and Python paths")
endif()
file(REMOVE_RECURSE "${TEST_BINARY_DIR}")
file(MAKE_DIRECTORY "${TEST_BINARY_DIR}/project with spaces")
file(COPY "${SOURCE_DIR}/tests/fixtures/functions/cross_module.th"
          "${SOURCE_DIR}/tests/fixtures/functions/layers.th"
          "${SOURCE_DIR}/tests/fixtures/functions/flattened.th"
     DESTINATION "${TEST_BINARY_DIR}/project with spaces")
set(entry "${TEST_BINARY_DIR}/project with spaces/cross_module.th")

function(run_ok label)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${label} failed: ${output}${error}")
    endif()
endfunction()
function(run_bad name source expected)
    file(WRITE "${TEST_BINARY_DIR}/${name}.th" "${source}")
    execute_process(COMMAND "${THIRAN_EXECUTABLE}" --plan "${TEST_BINARY_DIR}/${name}.th"
                    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(result EQUAL 0 OR NOT output MATCHES "${expected}")
        message(FATAL_ERROR "${name} expected ${expected}: ${output}${error}")
    endif()
endfunction()

run_ok("local function plan" "${THIRAN_EXECUTABLE}" --plan "${SOURCE_DIR}/tests/fixtures/functions/local.th")
execute_process(COMMAND "${THIRAN_EXECUTABLE}" --plan "${entry}"
                RESULT_VARIABLE result OUTPUT_VARIABLE first_plan ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "cross-module plan failed: ${first_plan}${error}")
endif()
execute_process(COMMAND "${THIRAN_EXECUTABLE}" --plan "${entry}" WORKING_DIRECTORY "/tmp"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT "${output}" STREQUAL "${first_plan}")
    message(FATAL_ERROR "function plan is not cwd-independent and deterministic: ${output}${error}")
endif()
run_ok("identity plan" "${THIRAN_EXECUTABLE}" --plan "${SOURCE_DIR}/tests/fixtures/functions/identity.th")
run_ok("nested plan" "${THIRAN_EXECUTABLE}" --plan "${SOURCE_DIR}/tests/fixtures/functions/nested.th")
run_ok("emit plan" "${THIRAN_EXECUTABLE}" --emit-plan "${entry}" "${TEST_BINARY_DIR}/plan.txt")
run_ok("emit executor" "${THIRAN_EXECUTABLE}" --emit-region-executor "${entry}" "${TEST_BINARY_DIR}/function.py")
run_ok("emit flat executor" "${THIRAN_EXECUTABLE}" --emit-region-executor "${TEST_BINARY_DIR}/project with spaces/flattened.th" "${TEST_BINARY_DIR}/flat.py")
run_ok("executor import safety" "${PYTHON_EXECUTABLE}" -m py_compile "${TEST_BINARY_DIR}/function.py")
file(WRITE "${TEST_BINARY_DIR}/compare.py"
"import pathlib,subprocess,sys,torch\n"
"r=pathlib.Path(sys.argv[1]); p=sys.executable\n"
"x=torch.arange(4,dtype=torch.float32).reshape(2,2); torch.save({'X':x},r/'input.pt')\n"
"subprocess.run([p,str(r/'function.py'),'--run',str(r/'input.pt'),str(r/'function.pt')],check=True)\n"
"subprocess.run([p,str(r/'flat.py'),'--run',str(r/'input.pt'),str(r/'flat.pt')],check=True)\n"
"a=torch.load(r/'function.pt',map_location='cpu',weights_only=True); b=torch.load(r/'flat.pt',map_location='cpu',weights_only=True)\n"
"assert list(a)==list(b)==['O']; assert a['O'].shape==b['O'].shape and a['O'].dtype==b['O'].dtype and a['O'].device==b['O'].device\n"
"torch.testing.assert_close(a['O'],b['O'],rtol=1e-5,atol=1e-6)\n"
"expected=torch.relu(x@torch.ones((2,2))+torch.ones((2,2))); torch.testing.assert_close(a['O'],expected,rtol=1e-5,atol=1e-6)\n")
run_ok("numerical equivalence" "${PYTHON_EXECUTABLE}" "${TEST_BINARY_DIR}/compare.py" "${TEST_BINARY_DIR}")

run_bad("arity" "fn f(x) {\n return x\n}\nX=Input()\nY=f()\n" "expects 1 arguments")
run_bad("unresolved" "X=Input()\nY=missing(X)\n" "unresolved function")
run_bad("capture" "fn f(x) {\n y=Add(x,C)\n return y\n}\nC=Constant(1)\nX=Input()\n" "cannot capture module value")
execute_process(COMMAND "${THIRAN_EXECUTABLE}" --plan "${SOURCE_DIR}/tests/fixtures/functions/recursive.th"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT output MATCHES "recursive function cycle")
    message(FATAL_ERROR "recursion diagnostic missing: ${output}${error}")
endif()

set(prefix "${TEST_BINARY_DIR}/prefix")
run_ok("install" "${CMAKE_COMMAND}" --install "${BINARY_DIR}" --prefix "${prefix}")
run_ok("installed function plan" "${prefix}/bin/thiran" --plan "${entry}")
foreach(example cnn mlp)
    run_ok("legacy ${example}" "${THIRAN_EXECUTABLE}" --plan "${SOURCE_DIR}/examples/${example}.th")
endforeach()
