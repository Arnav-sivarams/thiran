cmake_minimum_required(VERSION 3.20)
file(REMOVE_RECURSE "${TEST_BINARY_DIR}")
file(MAKE_DIRECTORY "${TEST_BINARY_DIR}/project with spaces")
file(COPY "${SOURCE_DIR}/tests/fixtures/modules/basic/main.th"
          "${SOURCE_DIR}/tests/fixtures/modules/basic/weights.th"
          "${SOURCE_DIR}/tests/fixtures/modules/basic/flattened.th"
     DESTINATION "${TEST_BINARY_DIR}/project with spaces")
set(entry "${TEST_BINARY_DIR}/project with spaces/main.th")
foreach(repository_artifact generated.py graph.dot)
    if(EXISTS "${SOURCE_DIR}/${repository_artifact}")
        set(${repository_artifact}_existed TRUE)
        file(SHA256 "${SOURCE_DIR}/${repository_artifact}" ${repository_artifact}_sha)
    else()
        set(${repository_artifact}_existed FALSE)
    endif()
endforeach()

function(require_success label)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${label} failed: ${error}${output}")
    endif()
endfunction()
function(run_failure file expected)
    execute_process(COMMAND "${THIRAN_EXECUTABLE}" --plan "${file}"
                    WORKING_DIRECTORY "${TEST_BINARY_DIR}"
                    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(result EQUAL 0 OR NOT output MATCHES "${expected}")
        message(FATAL_ERROR "expected failure ${expected}: ${output}${error}")
    endif()
endfunction()

execute_process(COMMAND "${THIRAN_EXECUTABLE}" "${entry}"
                WORKING_DIRECTORY "${TEST_BINARY_DIR}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_success("normal module compilation")
if(NOT EXISTS "${TEST_BINARY_DIR}/generated.py" OR NOT EXISTS "${TEST_BINARY_DIR}/graph.dot")
    message(FATAL_ERROR "normal module artifacts missing from isolated directory")
endif()

execute_process(COMMAND "${THIRAN_EXECUTABLE}" --plan "${entry}"
                WORKING_DIRECTORY "${TEST_BINARY_DIR}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_success("plan")
set(first_plan "${output}")
execute_process(COMMAND "${THIRAN_EXECUTABLE}" --plan "${entry}"
                WORKING_DIRECTORY "/tmp"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_success("cwd-independent plan")
if(NOT "${output}" STREQUAL "${first_plan}")
    message(FATAL_ERROR "module plan is nondeterministic")
endif()
execute_process(COMMAND "${THIRAN_EXECUTABLE}" --emit-plan "${entry}" "${TEST_BINARY_DIR}/plan.txt"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_success("emit plan")
execute_process(COMMAND "${THIRAN_EXECUTABLE}" --emit-region-executor "${entry}" "${TEST_BINARY_DIR}/executor.py"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_success("emit executor")
if(NOT EXISTS "${TEST_BINARY_DIR}/plan.txt" OR NOT EXISTS "${TEST_BINARY_DIR}/executor.py")
    message(FATAL_ERROR "module artifacts missing")
endif()
file(SIZE "${TEST_BINARY_DIR}/executor.py" executor_size)
if(executor_size EQUAL 0)
    message(FATAL_ERROR "module executor empty")
endif()
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -m py_compile "${TEST_BINARY_DIR}/executor.py"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_success("executor import safety")
execute_process(COMMAND "${THIRAN_EXECUTABLE}" --emit-region-executor
                        "${TEST_BINARY_DIR}/project with spaces/flattened.th" "${TEST_BINARY_DIR}/flat-executor.py"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_success("flattened executor")
file(WRITE "${TEST_BINARY_DIR}/compare.py"
"import pathlib, subprocess, sys, torch\n"
"root=pathlib.Path(sys.argv[1]); py=sys.executable\n"
"inputs={'X': torch.arange(4, dtype=torch.float32).reshape(2,2)}\n"
"torch.save(inputs, root/'inputs.pt')\n"
"subprocess.run([py,str(root/'executor.py'),'--run',str(root/'inputs.pt'),str(root/'module.pt')],check=True)\n"
"subprocess.run([py,str(root/'flat-executor.py'),'--run',str(root/'inputs.pt'),str(root/'flat.pt')],check=True)\n"
"a=torch.load(root/'module.pt',map_location='cpu',weights_only=True)\n"
"b=torch.load(root/'flat.pt',map_location='cpu',weights_only=True)\n"
"assert list(a)==list(b)==['O']; torch.testing.assert_close(a['O'],b['O'],rtol=1e-5,atol=1e-6)\n")
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${TEST_BINARY_DIR}/compare.py" "${TEST_BINARY_DIR}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_success("module numerical equivalence")

file(WRITE "${TEST_BINARY_DIR}/missing.th" "import \"no.th\" as n\nX = Input()\n")
run_failure("${TEST_BINARY_DIR}/missing.th" "could not resolve import")
file(WRITE "${TEST_BINARY_DIR}/absolute.th" "import \"/tmp/no.th\" as n\nX = Input()\n")
run_failure("${TEST_BINARY_DIR}/absolute.th" "absolute import")
file(WRITE "${TEST_BINARY_DIR}/escape.th" "import \"../outside.th\" as n\nX = Input()\n")
run_failure("${TEST_BINARY_DIR}/escape.th" "escapes project root")
file(MAKE_DIRECTORY "${TEST_BINARY_DIR}/directory.th")
file(WRITE "${TEST_BINARY_DIR}/directory-main.th" "import \"directory.th\" as d\nX = Input()\n")
run_failure("${TEST_BINARY_DIR}/directory-main.th" "not a regular file")
file(WRITE "${TEST_BINARY_DIR}/late.th" "X = Input()\nimport \"weights.th\" as w\n")
run_failure("${TEST_BINARY_DIR}/late.th" "imports must precede")
file(WRITE "${TEST_BINARY_DIR}/private-lib.th" "P = Constant(1)\n")
file(WRITE "${TEST_BINARY_DIR}/private-main.th" "import \"private-lib.th\" as p\nX = Input()\nY = Add(X, p.P)\nO = Output(Y)\n")
run_failure("${TEST_BINARY_DIR}/private-main.th" "not exported")
file(WRITE "${TEST_BINARY_DIR}/cycle-a.th" "import \"cycle-b.th\" as b\nexport A = Constant(1)\n")
file(WRITE "${TEST_BINARY_DIR}/cycle-b.th" "import \"cycle-a.th\" as a\nexport B = Constant(1)\n")
run_failure("${TEST_BINARY_DIR}/cycle-a.th" "import cycle")
file(WRITE "${TEST_BINARY_DIR}/leaf.th" "export C = Constant(1)\n")
file(WRITE "${TEST_BINARY_DIR}/left.th" "import \"leaf.th\" as leaf\nexport V = ReLU(leaf.C)\n")
file(WRITE "${TEST_BINARY_DIR}/right.th" "import \"leaf.th\" as leaf\nexport V = Sigmoid(leaf.C)\n")
file(WRITE "${TEST_BINARY_DIR}/diamond.th" "import \"left.th\" as left\nimport \"right.th\" as right\nX = Input()\nA = Add(X, left.V)\nB = Add(A, right.V)\nO = Output(B)\n")
execute_process(COMMAND "${THIRAN_EXECUTABLE}" --plan "${TEST_BINARY_DIR}/diamond.th"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_success("nested diamond modules with same exported names")
file(WRITE "${TEST_BINARY_DIR}/self.th" "import \"self.th\" as self\nX = Input()\n")
run_failure("${TEST_BINARY_DIR}/self.th" "imports itself")
file(WRITE "${TEST_BINARY_DIR}/unresolved-alias.th" "X = Input()\nY = ReLU(nope.X)\nO = Output(Y)\n")
run_failure("${TEST_BINARY_DIR}/unresolved-alias.th" "unresolved import alias")
file(WRITE "${TEST_BINARY_DIR}/export-lib.th" "export C = Constant(1)\n")
file(WRITE "${TEST_BINARY_DIR}/unresolved-symbol.th" "import \"export-lib.th\" as lib\nX = Input()\nY = Add(X, lib.Nope)\nO = Output(Y)\n")
run_failure("${TEST_BINARY_DIR}/unresolved-symbol.th" "not exported")
file(WRITE "${TEST_BINARY_DIR}/input-lib.th" "export X = Input()\n")
file(WRITE "${TEST_BINARY_DIR}/duplicate-alias.th" "import \"private-lib.th\" as p\nimport \"input-lib.th\" as p\nX = Input()\n")
run_failure("${TEST_BINARY_DIR}/duplicate-alias.th" "duplicate import alias")
file(WRITE "${TEST_BINARY_DIR}/input-main.th" "import \"input-lib.th\" as i\nX = Input()\n")
run_failure("${TEST_BINARY_DIR}/input-main.th" "Input is only allowed")
file(WRITE "${TEST_BINARY_DIR}/output-lib.th" "C = Constant(1)\nexport O = Output(C)\n")
file(WRITE "${TEST_BINARY_DIR}/output-main.th" "import \"output-lib.th\" as out\nX = Input()\n")
run_failure("${TEST_BINARY_DIR}/output-main.th" "Output is only allowed")
file(WRITE "${TEST_BINARY_DIR}/reserved.th" "__thiran_module_x = Input()\n")
run_failure("${TEST_BINARY_DIR}/reserved.th" "reserved internal prefix")
file(COPY "${TEST_BINARY_DIR}/project with spaces/weights.th" DESTINATION "${TEST_BINARY_DIR}/space-import")
file(RENAME "${TEST_BINARY_DIR}/space-import/weights.th" "${TEST_BINARY_DIR}/space-import/weight values.th")
file(WRITE "${TEST_BINARY_DIR}/space-import/main.th" "import \"weight values.th\" as weights\nX = Input(2, 2)\nY = Add(X, weights.B)\nO = Output(Y)\n")
execute_process(COMMAND "${THIRAN_EXECUTABLE}" --plan "${TEST_BINARY_DIR}/space-import/main.th"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_success("import path containing spaces")

set(prefix "${TEST_BINARY_DIR}/prefix")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BINARY_DIR}" --prefix "${prefix}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_success("module install")
execute_process(COMMAND "${prefix}/bin/thiran" --plan "${entry}"
                WORKING_DIRECTORY "/tmp"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_success("installed module plan")

foreach(example cnn mlp simple transformer)
    execute_process(COMMAND "${THIRAN_EXECUTABLE}" --plan "${SOURCE_DIR}/examples/${example}.th"
                    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    require_success("legacy ${example}")
endforeach()

foreach(repository_artifact generated.py graph.dot)
    if(${repository_artifact}_existed)
        file(SHA256 "${SOURCE_DIR}/${repository_artifact}" final_sha)
        if(NOT "${final_sha}" STREQUAL "${${repository_artifact}_sha}")
            message(FATAL_ERROR "repository ${repository_artifact} changed")
        endif()
    elseif(EXISTS "${SOURCE_DIR}/${repository_artifact}")
        message(FATAL_ERROR "repository ${repository_artifact} was created")
    endif()
endforeach()
