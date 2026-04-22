import { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';

export default function ChartWidget({ data, atr, atrMultiplier }) {
  const chartContainerRef = useRef();
  const chartRef = useRef();

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: { background: { type: 'solid', color: '#06090f' }, textColor: '#8b949e' },
      grid: { vertLines: { visible: false }, horzLines: { color: '#1c2128' } },
      rightPriceScale: { borderColor: '#30363d' },
      timeScale: { borderColor: '#30363d', timeVisible: true },
      height: 500,
    });
    
    chartRef.current = chart;

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#3fb950', downColor: '#f85149',
      borderVisible: false, wickUpColor: '#3fb950', wickDownColor: '#f85149'
    });

    const emaSeries = chart.addLineSeries({
      color: '#58a6ff', lineWidth: 1, title: 'EMA 20'
    });

    const tpSeries = chart.addLineSeries({
      color: '#3fb950', lineWidth: 1, lineStyle: 2, title: 'TP'
    });

    const slSeries = chart.addLineSeries({
      color: '#f85149', lineWidth: 1, lineStyle: 2, title: 'SL'
    });

    if (data?.candles?.length > 0) {
      candleSeries.setData(data.candles);
      
      if (data.markers && data.markers.length > 0) {
        candleSeries.setMarkers(data.markers);
      }
      
      if (data.ema) {
         emaSeries.setData(data.ema);
      }
      
      if (atr && atrMultiplier && data.candles.length > 0) {
        const lastPrice = data.candles[data.candles.length - 1].close;
        const dist = atr * atrMultiplier;
        
        const tpData = data.candles.slice(-15).map(c => ({ time: c.time, value: lastPrice + dist }));
        const slData = data.candles.slice(-15).map(c => ({ time: c.time, value: lastPrice - dist }));
        
        tpSeries.setData(tpData);
        slSeries.setData(slData);
      }
    }

    return () => chart.remove();
  }, [data, atr, atrMultiplier]);

  return <div ref={chartContainerRef} className="w-full h-full" />;
}
