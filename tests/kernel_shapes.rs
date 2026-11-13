use std::io::Write;
use std::process::Command;

fn write_fixture(path: &std::path::Path) {
    let json = r#"{"traceEvents":[
        {"ph":"X","cat":"cpu_op","name":"aten::mm","pid":1,"tid":1,"ts":1.0,"dur":5.0,
         "args":{"External id":7,"Input Dims":[[16,64],[64,32]],"Input type":["float","float"],"flops":65536}},
        {"ph":"X","cat":"cuda_runtime","name":"cudaLaunchKernel","pid":1,"tid":1,"ts":2.0,"dur":1.0,
         "args":{"External id":7,"correlation":42}},
        {"name":"gemm","cat":"kernel","ph":"X","ts":100.0,"dur":5.0,"pid":0,"tid":4,
         "args":{"device":0,"stream":4,"correlation":42}},
        {"name":"gemm","cat":"kernel","ph":"X","ts":200.0,"dur":7.0,"pid":0,"tid":4,
         "args":{"device":0,"stream":4,"correlation":43}},
        {"name":"add","cat":"kernel","ph":"X","ts":400.0,"dur":1.0,"pid":0,"tid":4,
         "args":{"device":0,"stream":4}}
    ]}"#;
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(json.as_bytes()).unwrap();
}

#[test]
fn kernel_shapes_prints_ranked_csv() {
    let dir = std::env::temp_dir().join(format!("ptt-shapes-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let trace = dir.join("fixture.json");
    write_fixture(&trace);

    let run = |extra: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_pytorch-trace-tui"))
            .arg("--kernel-shapes")
            .args(extra)
            .arg(&trace)
            .output()
            .expect("binary runs");
        assert!(out.status.success(), "exit ok: {:?}", out.status);
        (
            String::from_utf8(out.stdout).unwrap(),
            String::from_utf8(out.stderr).unwrap(),
        )
    };

    let (stdout, stderr) = run(&[]);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines,
        vec![
            "kernel,op,calls,total_us,mean_us,input_dims,input_types,flops_per_call",
            "gemm,,1,7.0,7.00,,,",
            r#"gemm,aten::mm,1,5.0,5.00,"[[16,64],[64,32]]","[""float"",""float""]",65536"#,
            "add,,1,1.0,1.00,,,",
        ]
    );
    assert!(
        stderr.contains("3 kernels, 2 without a launching operator"),
        "{stderr}"
    );

    let (stdout, _) = run(&["--name", "gem", "--top", "1"]);
    assert_eq!(stdout.lines().count(), 2, "header + top 1:\n{stdout}");

    std::fs::remove_dir_all(&dir).ok();
}
