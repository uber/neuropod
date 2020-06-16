package org.neuropod;

import java.util.*;

public class TestMain {

    public static void main(String[] args) throws Exception {
        System.out.println("Start working......");

        String modelPath = "src/test/resources/test.zip";
        System.out.println("Loading model = " + System.getProperty("user.dir") + modelPath);
        // Load a neuropod model
        Neuropod neuropod = new Neuropod(modelPath);
        Map<String, Object> inputs = new HashMap<>();

        List<Float> elements1 = new ArrayList<>();
        elements1.add(1.0f);
        inputs.put("request_location_latitude",elements1);

        List<Float> elements2 = new ArrayList<>();
        elements2.add(2.0f);
        inputs.put("request_location_longitude",elements2);

        System.out.println("The input is");
        System.out.println(inputs);



        NeuropodValueMap valueMap = neuropod.infer(inputs);

        List<String> keyList = valueMap.getKeyList();

        System.out.println("The output is");
        for (String key : keyList) {
            System.out.print(key + "=");
            NeuropodValue value = valueMap.getValue(key);
            List<Object> res = value.toList();
            System.out.println(res);
        }
        neuropod.close();
        valueMap.close();
    }

}
